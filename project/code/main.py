import argparse
import main_utils
import logger

_logger = logger.get_logger(__name__)

PHASES = {'base results':
              {'get_results_func': main_utils.get_base_results,
               'store_results_to_file_func': main_utils.store_base_results,
               'extract_results_from_file_func': main_utils.extract_base_results,
               'file_name': 'base_results',
               'run': True},

          'low shot results':
              {'get_results_func': main_utils.get_low_shot_results,
               'store_results_to_file_func': main_utils.store_low_shot_results,
               'extract_results_from_file_func': main_utils.extract_low_shot_results,
               'file_name': 'low_shot_results',
               'run': True},

          'improved low shot results':
              {'get_results_func': main_utils.get_improved_low_shot_results,
               'store_results_to_file_func': main_utils.store_improved_low_shot_results,
               'extract_results_from_file_func': main_utils.extract_improved_low_shot_results,
               'file_name': 'low_shot_results',
               'run': True}
          }


def run(n_files):
    main_utils.create_sub_directory()
    raw_data = main_utils.get_raw_data(n_files)
    results = {}
    for phase in PHASES.keys():
        phase_params = PHASES[phase]
        if phase_params['run']:
            _logger.info('run phase - %s' % phase)
            results[phase] = phase_params['get_results_func'](raw_data)
            _logger.info('store phase %s result to %s' % (phase, phase_params['file_name']))
            phase_params['store_results_to_file_func'](results[phase], phase_params['file_name'])
        else:
            _logger.info('store phase %s from %s' % (phase, phase_params['file_name']))
            results[phase] = phase_params['extarct_results_from_file_func'](phase_params['file_name'])

    main_utils.parse_results(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--n_files', help='number of files to process', type=int, default=1)
    args = parser.parse_args()

    run(args.n_files)
