import argparse
from subprocess import call

from config import *


def main(disease, feature, n_epochs, remote, remote_update):
    if not remote:
        import main as _main_
        return _main_.main(disease, feature, n_epochs, remote)

    call('gcloud compute instances start --zone={zone} {vm}'.format(zone=gcloud_zone, vm=gcloud_vm), shell=True)

    if remote_update:
        print('Updating code.')
        call('gcloud compute ssh --zone={zone} {vm} --command "rm -rfv {remote_dir}/*"'.format(
            zone=gcloud_zone,
            vm=gcloud_vm,
            remote_dir=gcloud_code_dir
        ), shell=True)
        call('gcloud compute scp --recurse --compress --zone={zone} {local_dir} {username}@{vm}:~/nlprun/'.format(
            zone=gcloud_zone,
            local_dir=local_code_dir,
            username=gcloud_username,
            vm=gcloud_vm
        ), shell=True)
        print('done.')

    print('Starting experiment remotely.')
    call('gcloud compute ssh --zone={zone} {vm} --command "nohup {python_path} {remote_dir}/main.py {disease} -r -e {epochs} > run.out 2> run.err < /dev/null &" &'.format(
        zone=gcloud_zone,
        vm=gcloud_vm,
        remote_dir=gcloud_code_dir,
        disease=disease,
        epochs=n_epochs,
        python_path=gcloud_python_path if gcloud_python_path else 'python'
    ), shell=True)
    print('done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('diease', help='disease to experiment on')
    parser.add_argument('-e', '--n_epochs', help='number of epochs to run', type=int, default=10)
    parser.add_argument('-r', '--remote', help='run remotely on the configured gcloud vm', action='store_true')
    parser.add_argument('-u', '--update', help='update the remote code', action='store_true')

    args = parser.parse_args()

    main(args.disease, args.n_epochs, args.remote, args.update)

