from os.path import join, commonprefix
from getpass import getuser
from time import time
from string import ascii_uppercase, digits
from math import ceil
from multiprocessing import Pool

from tqdm import tqdm

from ..os import call, makedirs
from .. import const
from ..interact import ask_to_proceed
from ..log import get_logger
logger = get_logger()


class Launcher():
    def __init__(
            self, label, print_instead=False, borg_submitters=24,
            borg_user=None, borg_cell=None, borg_priority=None, gfs_user=None):
        self.label = label
        self.pkg_bin = self._derive_bin()
        self.print_instead = print_instead
        self.myself = getuser()
        if borg_user is None:
            borg_user = self.myself
        self.borg_user = borg_user
        if gfs_user is None:
            gfs_user = self.myself
        self.gfs_user = gfs_user
        self.borg_submitters = borg_submitters
        # Deriving other values
        if borg_priority is None:
            self.priority = self._select_priority()
        else:
            self.priority = borg_priority
        if borg_cell is None:
            self.cell = self._select_cell()
        else:
            self.cell = borg_cell

    def _derive_bin(self):
        assert ':' in self.label, "Must specify target explicitly"
        pkg_bin = self.label.split(':')[-1]
        if pkg_bin.endswith('_mpm'):
            pkg_bin = pkg_bin[:-4]
        logger.warning(("Package binary derived to be `%s`, so make sure "
                        "BUILD is consistent with this"), pkg_bin)
        return pkg_bin

    def _select_priority(self):
        if self.borg_user == self.myself:
            return 0
        return 115

    def _select_cell(self):
        if self.borg_user == self.myself:
            cell = 'qu'
        elif self.borg_user == 'gcam-eng':
            cell = 'ok'
        elif self.borg_user == 'gcam-gpu':
            cell = 'is'
        else:
            raise NotImplementedError(self.borg_user)
        return cell

    def blaze_run(self, blaze_dict=None, param_dict=None):
        bash_cmd = 'blaze run -c opt %s' % self.label
        # Blaze parameters
        if blaze_dict is not None:
            for k, v in blaze_dict.items():
                bash_cmd += ' --{flag}={val}'.format(flag=k, val=v)
        # Job parameters
        if param_dict is not None:
            bash_cmd += ' --'
            for k, v in param_dict.items():
                bash_cmd += ' --{flag}={val}'.format(flag=k, val=v)
        # To avoid IO permission issues
        bash_cmd += ' --gfs_user=%s' % self.gfs_user
        # To use my own account for Bigstore
        bash_cmd += ' --bigstore_anonymous'
        if self.print_instead:
            logger.info("To blaze-run the job, run:\n\t%s\n", bash_cmd)
        else:
            call(bash_cmd)
            # FIXME: sometimes stdout can't catch the printouts (e.g., tqdm)

    def build_for_borg(self):
        assert self.label.endswith('_mpm'), \
            "Label must be MPM, because .borg generation assumes temporary MPM"
        bash_cmd = 'rabbit build -c opt %s' % self.label
        # FIXME: --verifiable leads to "MPM failed to find the .pkgdef file"
        if self.print_instead:
            logger.info("To build for Borg, run:\n\t%s\n", bash_cmd)
        else:
            retcode, _, _ = call(bash_cmd)
            assert retcode == 0, "Build failed"

    @staticmethod
    def _divide_jobs(job_ids, param_dicts, n_tasks_per_job):
        n_jobs = ceil(len(job_ids) / n_tasks_per_job)
        # Get shared
        shared_params = dict(
            set.intersection(*(set(d.items()) for d in param_dicts)))
        shared_id = commonprefix(job_ids)
        # Collect task-specific parameters for each job
        job_names, job_specific_params, job_n_tasks = [], [], []
        for job_i in range(n_jobs):
            start = job_i * n_tasks_per_job
            end = min(start + n_tasks_per_job, len(param_dicts))
            task_specific_params = {}
            for task_params in param_dicts[start:end]:
                for k, v in task_params.items():
                    if k not in shared_params: # not shared
                        if k not in task_specific_params:
                            task_specific_params[k] = []
                        task_specific_params[k].append(v)
            # Sanity check: different parameters should say the same number
            # of tasks
            n_tasks = [len(v) for _, v in task_specific_params.items()]
            assert len(set(n_tasks)) == 1, \
                "Value lists for task parameters must be of the same length"
            job_n_tasks.append(n_tasks[0])
            # These tasks constitute one job
            job_specific_params.append(task_specific_params)
            job_names.append(shared_id + '_job%04d' % job_i)
        return job_names, shared_params, job_specific_params, job_n_tasks

    def submit_to_borg(self, job_ids, param_dicts, n_tasks_per_job=1):
        assert isinstance(job_ids, list) and isinstance(param_dicts, list), \
            "If submitting just one job, make both arguments single-item lists"
        assert len(job_ids) == len(param_dicts), \
            "Numbers of job IDs and parameter dictionaries must be equal"
        assert job_ids, "No jobs"
        # Divide tasks into jobs
        if len(job_ids) == 1:
            job_names = job_ids
            shared_params = param_dicts[0]
            job_specific_params = [{}]
            job_n_tasks = [0]
        else:
            job_names, shared_params, job_specific_params, job_n_tasks = \
                self._divide_jobs(job_ids, param_dicts, n_tasks_per_job)
        n_jobs = len(job_names)
        # Ask for confirmation
        ask_to_proceed(
            ("About to submit {n} jobs:\n\t{names},\nwith constant parameters:"
             "\n\t{shared}").format(
                 n=n_jobs, names=job_names, shared=shared_params))
        # Sequential submissions, if just one job, or no parallel workers,
        # or printing only
        if n_jobs == 1 or self.borg_submitters == 0 or self.print_instead:
            for job_name, job_specific, n_tasks in tqdm(
                    zip(job_names, job_specific_params, job_n_tasks),
                    desc="Submitting jobs to Borg", total=n_jobs):
                self._borg_run((job_name, shared_params, job_specific, n_tasks))
        # Parallel submissions
        else:
            pool = Pool(self.borg_submitters)
            list(tqdm(pool.imap_unordered(
                self._borg_run,
                [(x, shared_params, y, z) for x, y, z in
                 zip(job_names, job_specific_params, job_n_tasks)]
            ), total=len(job_names), desc="Submitting jobs to Borg"))
            pool.close()
            pool.join()

    def _borg_run(self, args):
        """.borg generation assumed temporary MPM.
        """
        job_name, shared_params, task_params, n_tasks = args
        borg_f = self.__gen_borg_file(
            job_name, shared_params, task_params, n_tasks)
        # Submit
        action = 'reload'
        # NOTE: runlocal doesn't work for temporary MPM: b/74472376
        bash_cmd = ('borgcfg {f} {a} --borguser {u} --skip_confirmation '
                    '--strict_confirmation_threshold=65536').format(
                        f=borg_f, a=action, u=self.borg_user)
        if self.print_instead:
            logger.info("To launch the job on Borg, run:\n\t%s\n", bash_cmd)
        else:
            call(bash_cmd)

    def __gen_borg_file(self, job_name, shared_params, task_params, n_tasks):
        borg_file_str = self.___format_borg_file_str(
            job_name, shared_params, task_params, n_tasks)
        out_dir = join(const.Dir.tmp, '{t}_{s}'.format(s=_random_str(16),
                                                       t=time()))
        makedirs(out_dir)
        borg_f = join(out_dir, '%s.borg' % job_name)
        with open(borg_f, 'w') as h:
            h.write(borg_file_str)
        logger.info("Generated .borg at\n\t%s", borg_f)
        return borg_f

    @staticmethod
    def ____to_str(v):
        if isinstance(v, str):
            v = "'%s'" % v
        elif isinstance(v, int):
            v = "%d" % v
        elif isinstance(v, float):
            v = "%f" % v
        elif isinstance(v, list):
            v = "'" + ",".join(str(x) for x in v) + "'"
        else:
            raise TypeError(type(v))
        return v

    def ___format_borg_file_str(
            self, job_name, shared_params, task_params, n_tasks):
        tab = ' ' * 4
        file_str = '''job {job_name} = {{
    // What cell should we run in?
    runtime = {{
        // 'oregon' // automatically picks a Borg cell with free capacity
        cell = '{cell}',
    }}

    // What packages are needed?
    packages {{
        package bin = {{
            // A blaze label pointing to a `genmpm(temporal=1)` rule. Borgcfg will
            // build a "temporal MPM" on the fly out of files in the blaze-genfiles
            // directory. See go/temporal-mpm for full documentation.
            blaze_label = '{label}',
        }}
    }}

    // What program are we going to run?
    package_binary = 'bin/{binary}'

    // What command line parameters should we pass to this program?
    args = {{
    '''.format(job_name=job_name, cell=self.cell,
               label=self.label, binary=self.pkg_bin)
        # Add shared parameters
        for k, v in shared_params.items():
            v = self.____to_str(v)
            str_ = '{tab}{tab}{flag} = {val},\n'
            file_str += str_.format(tab=tab, flag=k, val=v)
        # Add placeholders for task-specific parameters
        for k, _ in task_params.items():
            v = "'%{}%'".format(k)
            file_str += '{tab}{tab}{flag} = {val},\n'.format(
                tab=tab, flag=k, val=v)
        file_str += '{tab}}}'.format(tab=tab)
        # Fill in task-specific parameters, if any
        if n_tasks > 0:
            file_str += '''

    replicas = {replicas}

    task_args {{'''.format(replicas=n_tasks)
            for k, vlist in task_params.items():
                file_str += '\n{tab}{tab}{flag} = {val_list}'.format(
                    tab=tab, flag=k, val_list=vlist)
            # The rest
            file_str += '''
    }'''
        file_str += '''

    // What resources does this program need to run?
    requirements = {{
        autopilot = true,
        autopilot_params {{
            // Let autopilot increase limits past the Borg pickiness limit
            // scheduling_strategy = 'NO_SCHEDULING_SLO',
            // Resources are capped to the Borg pickiness limit
            scheduling_strategy = 'FAST_SCHEDULING',
        }}
        // ram = 1024M,
        // use_ram_soft_limit = true,
        // local_ram_fs_dir {{ d1 = {{ size = 4096M }} }},
        // cpu = 12,
    }}

    // How latency-sensitive is this program?
    appclass = {{
        type = 'LATENCY_TOLERANT_SECONDARY',
    }}

    permissions = {{
        user = '{user}',
    }}

    scheduling = {{
        max_task_failures = -1
        max_dead_tasks = -1

        priority = {priority},
    '''.format(user=self.borg_user, priority=self.priority)
        if self.priority == 115:
            file_str += '''
        batch_quota {
            strategy = 'RUN_SOON'
        }
    }
}'''
        else:
            file_str += '''}
}'''
        return file_str


def _random_str(l):
    from random import choices # requires >= 3.6

    return ''.join(choices(ascii_uppercase + digits, k=l))
