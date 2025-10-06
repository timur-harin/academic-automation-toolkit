"""
To understand this file either contact me https://t.me/unaimillan (or @system205) or
heavily use debugger to understand the structure of the dictionaries
"""
import hashlib
import json
import logging
import os
import os.path
import re
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from dotenv import load_dotenv

import codeforces_api # Need to add {'asManager': 'true'} in the 'parameters' variable inside the library

CONTEST_ID = 639210
END_DATE=datetime(year=2025, month=10, day=4, hour=23, minute=59)
ALL_PROBLEM_INDEXES = 'A B'
CF_GROUP_ID = '8332GXpGlQ'
LATE_SUBMISSION_IDS = []
INSTRUCTOR_HANDLES = ['ThatDude', 'system205', 'Damurka', 'timur_harin', 'fizruk', 'spaghetti_coder']
MOODLE_CF_HANDLES_FILE = Path('./data/student-cf-handles.csv')
MOODLE_COURSE_PARTICIPANTS_FILE = Path('./data/course_participants.csv')
CF_HANDLES_SPREADSHEET_FILE = Path('./data/cf_handles.csv')
JPLAG_JAR_FILENAME = 'jplag-6.2.0.jar'
CF_SUBMISSION_URL = f'https://codeforces.com/group/{CF_GROUP_ID}/contest/{CONTEST_ID}/submission/'
AVG_SIMILARITY_SCORE = 0.6

# ------------------------------------------------------------------------------

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

SUBMISSIONS_SUPPORTED_FILE_EXTS = (".py", ".py3", ".pypy2", ".pypy3-64", ".pypy3")
ACCEPTED_LANGUAGES = ('PyPy 3', 'PyPy 2', 'Python 3', 'PyPy 3-64', 'Python 2')
SOLUTION_FILE_EXT = '.py'
load_dotenv('./.env')

# Generate api key and secret at https://codeforces.com/settings/api
api_key = os.getenv("CODEFORCES_API_KEY")
api_secret = os.getenv("CODEFORCES_API_SECRET")

if api_secret is None:
    print("Please, define appropriate env var with api secret")
    exit(-1)

cf_api = codeforces_api.CodeforcesApi(api_key, api_secret)
cf_parser = codeforces_api.CodeforcesParser()

DATA_DIR = Path('./data')
DATA_DIR.mkdir(exist_ok=True)

SUBMISSIONS_DIR = DATA_DIR / str(CONTEST_ID)
SOLUTIONS_DIR = DATA_DIR / 'Solutions'
JPLAG_WORK_DIR = DATA_DIR / 'JPlag'
JPLAG_WORK_DIR.mkdir(exist_ok=True)


@dataclass
class ShortSubmission:
    creation_time: datetime
    problem_index: str
    handle: str
    participant_type: str
    passed_test_count: str
    points: str
    verdict: str


def rename_submission_file_extensions():
    for path in list(SUBMISSIONS_DIR.iterdir()):
        if path.suffix != SOLUTION_FILE_EXT:
            if path.suffix in SUBMISSIONS_SUPPORTED_FILE_EXTS:
                shutil.move(path, path.with_suffix(SOLUTION_FILE_EXT))
                #print(f"Renaming {path}")
            else:
                os.remove(path)


def get_submission_path(submission_id: int) -> Path:
    return SUBMISSIONS_DIR / (str(submission_id) + SOLUTION_FILE_EXT)


def collect_handle_submission_per_task(user_solutions_per_task: dict[str, dict[str, int]]
                                       ) -> dict[str, list[tuple[str, int]]]:
    result = defaultdict(list)
    for handle, submissions in user_solutions_per_task.items():
        for task_idx, submission in submissions.items():
            result[task_idx].append((handle, submission))
    return result

def jplag_parse_results_file(result_file: Path):
    results = []
    fl_name_pattern = re.compile(r'(?P<st1>.*?)\.py-(?P<st2>.*?)\.py.json')

    with zipfile.ZipFile(result_file) as zpfile:
        result_names = list(
            filter(fl_name_pattern.fullmatch, zpfile.namelist()))
        # print(f'Detected {len(result_names)} files inside JPlag report. Expected number of files in folder is {len(result_names)+6}')

        for cur_result in result_names:
            cur_result_json = json.loads(zpfile.read(cur_result))

            st1_handle, task_id, subm1_id = cur_result_json['firstSubmissionId'].removesuffix(
                SOLUTION_FILE_EXT).rsplit('_', 2)
            st2_handle, task_id, subm2_id = cur_result_json['secondSubmissionId'].removesuffix(
                SOLUTION_FILE_EXT).rsplit('_', 2)

            if subm1_id == subm2_id:
                continue

            avg_sim = cur_result_json['similarities']['AVG']
            max_sim = cur_result_json['similarities']['MAX']
            longest_match = cur_result_json['similarities']['LONGEST_MATCH']
            max_len = cur_result_json['similarities']['MAXIMUM_LENGTH']

            total_tokens = sum(map(lambda x: len(x), cur_result_json['matches']))

            results.append({
                'Problem': task_id,
                'Handle1': st1_handle,
                'Handle2': st2_handle,
                'SubmitId1': subm1_id,
                'SubmitId2': subm2_id,
                'Link1': CF_SUBMISSION_URL + subm1_id,
                'Link2': CF_SUBMISSION_URL + subm2_id,
                'AvgSim': avg_sim,
                'MaxSim': max_sim,
                'LongestMatch': longest_match,
                'MaxLen': max_len
            })

            # print(json.dumps(cur_result_json, indent=2))
            # break
    return results


def jplag_measure_similarity(user_solutions_per_task: dict[str, dict[str, int]], cf_handles, participants_df):
    tasks_submissions = collect_handle_submission_per_task(user_solutions_per_task)

    tasks_to_consider = list(tasks_submissions.keys())
    result = []

    for task_idx in tasks_to_consider:
        task_path = JPLAG_WORK_DIR / task_idx
        task_path.mkdir(exist_ok=True)
        submission_paths = []

        for handle, submission in tasks_submissions[task_idx]:
            source_path = SUBMISSIONS_DIR / f'{submission}{SOLUTION_FILE_EXT}'
            dest_path = task_path / \
                f'{handle}_{task_idx}_{user_solutions_per_task[handle][task_idx]}{SOLUTION_FILE_EXT}'
            shutil.copy(source_path, dest_path)
            
            submission_paths.append(dest_path)

        print(f'Checking problem {task_idx} with {len(submission_paths)} submissions using JPlag')

        jp_res_name = f'{task_idx}_jp_results'
        jp_res_file = (JPLAG_WORK_DIR / jp_res_name).with_suffix('.jplag')

        def jplag_run_similarity_check():
            jp_proc = subprocess.run(f'cd {JPLAG_WORK_DIR} '
                                     f'&& java -jar ./{JPLAG_JAR_FILENAME} ./{task_idx} '
                                     f'-l python3 -M RUN -r {jp_res_name} -m 0.60',
                                     shell=True, capture_output=True)
            if jp_proc.returncode != 0:
                print('JPlag return code:', jp_proc.returncode)
                print(jp_proc.stderr)
                jp_proc.check_returncode()
        jplag_run_similarity_check()

        jp_results = jplag_parse_results_file(jp_res_file)
        result += jp_results

        print(f'Problem {task_idx} completed')

    # Filter JPlag results
    filt_result = list(filter(lambda x: x['AvgSim'] >= AVG_SIMILARITY_SCORE, result))
    print(f'Removing all cases with AVG similarity score less than {AVG_SIMILARITY_SCORE}')
    handle2name = {}
    for n,h in cf_handles.items():
        handle2name[h] = n
    for r in filt_result:
        try:
            r['Email1'] = participants_df[participants_df['Fullname'] == handle2name[r['Handle1']]]['Email address'].values[0] 
            r['Email2'] = participants_df[participants_df['Fullname'] == handle2name[r['Handle2']]]['Email address'].values[0] 
        except: continue
        
    if len(filt_result) > 0:
        pd.DataFrame(filt_result).to_csv(JPLAG_WORK_DIR / 'jplag_results.csv', index=False)


def calculate_jplag_metrics_and_statistics():
    jplag_res_df = pd.read_csv(
        JPLAG_WORK_DIR / 'jplag_results.csv', sep=',', header=0, index_col=False)

    total_match_per_handle = defaultdict(int)
    for _, row in jplag_res_df.iterrows():
        total_match_per_handle[row['Handle1']] += row['LongestMatch']
        total_match_per_handle[row['Handle2']] += row['LongestMatch']

    print("Top plagiarized matchers:")
    print(*list(map(lambda x: ' '.join(map(str, x)),
          sorted(list(total_match_per_handle.items()), key=lambda x: -x[1])))[:6], sep='\n')


def download_contest_status(contest: int, output: str) -> None:
    """
    Downloads all the contest submissions and stores them to text file

    :param contest: id of the contest
    :param output: file path to store results
    """
    submissions = cf_api.contest_status(contest)
    print(len(submissions), 'result(s) downloaded')
    json.dump(submissions, open(output, 'w'),
              default=lambda o: o.to_dict())


def load_contest_status(input_path) -> list[codeforces_api.Submission]:
    submissions = json.load(open(input_path))
    print(len(submissions), 'result(s) loaded')
    return submissions


def parse_submissions(submissions: list[json]) -> dict[int, ShortSubmission]:
    """
    Parses the list of submissions into the dictionary
    with key `submission id` and useful values

    :param submissions: list
    :return: dict
    """
    return {subm['id']: ShortSubmission(
        creation_time=datetime.fromtimestamp(subm['creation_time_seconds']),
        problem_index=subm['problem']['index'],
        handle=subm['author']['members'][0]['handle'],
        passed_test_count=subm['passed_test_count'],
        points=subm['points'],
        participant_type=subm['author']['participant_type'],
        verdict=subm['verdict'],
    )
        for subm in submissions}


ALL_PARTICIPANT_TYPES = (
    "CONTESTANT",
    "PRACTICE",
    "VIRTUAL",
    "MANAGER",
    "OUT_OF_COMPETITION",
)

ALL_SUBMISSION_VERDICTS = (
    "FAILED",
    "OK",
    "PARTIAL",
    "COMPILATION_ERROR",
    "RUNTIME_ERROR",
    "WRONG_ANSWER",
    "PRESENTATION_ERROR",
    "TIME_LIMIT_EXCEEDED",
    "MEMORY_LIMIT_EXCEEDED",
    "IDLENESS_LIMIT_EXCEEDED",
    "SECURITY_VIOLATED",
    "CRASHED",
    "INPUT_PREPARATION_CRASHED",
    "CHALLENGED",
    "SKIPPED",
    "TESTING",
    "REJECTED",
    ""
)


def filter_submissions(submissions: dict[int, ShortSubmission],
                       late_submission_idx: list[int],
                       submission_types: list[str] = ('CONTESTANT', 'VIRTUAL', 'MANAGER'),
                       submission_verdicts: list[str] = ALL_SUBMISSION_VERDICTS,
                       exclude_handles: list[str] = None,
                       problem_indexes: Iterable[str] = None,
                       start_time: datetime = None,
                       finish_time: datetime = None,
                       ) -> dict[int, ShortSubmission]:
    return {k: v for k, v in submissions.items() if
            (v.handle not in late_submission_idx)
            and (v.participant_type in submission_types)
            and (v.verdict in submission_verdicts)
            and (v.handle not in exclude_handles if exclude_handles else True)
            and (v.problem_index in problem_indexes if problem_indexes else True)
            and (start_time <= v.creation_time if start_time else True)
            and (v.creation_time <= finish_time if finish_time else True)
            }


def collect_submissions_per_user(submissions: dict[int, ShortSubmission]) -> dict[str, list[int]]:
    """
    Collect all submissions for each user

    :param submissions: dict
    :return: dict of user handle to list of submission ids
    """
    result = defaultdict(list)
    for subm_id, subm_data in submissions.items():
        result[subm_data.handle].append(subm_id)
    for handle in result:
        result[handle].sort()
    return result

def collect_last_user_submission_per_task(ordered_user_submissions: dict[str, list[int]],
                                             submissions_dict: dict[int,
                                                                    ShortSubmission]
                                             ) -> dict[str, dict[str, int]]:
    result = dict()
    for user, subs in ordered_user_submissions.items():
        user_solutions = dict()
        for sub in subs:
            subm_verdict = submissions_dict[sub].verdict
            subm_problem_idx = submissions_dict[sub].problem_index

            if subm_verdict == "OK" or subm_verdict == "PARTIAL":
                user_solutions[subm_problem_idx] = sub

        result[user] = user_solutions
    return result


def get_online_submission_count(contest):
    return len(cf_api.contest_status(contest))


def get_contest_problem_indexes(contest_id):
    return list(map(lambda v: v.index, cf_api.contest_standings(contest_id, count=1)['problems']))


def parse_course_participants(input_path: Path) -> pd.DataFrame:
    result = pd.read_csv(input_path, sep=',', header=0, index_col=False,
                         usecols=['First name', 'Last name', 'Email address'])
    result['Fullname'] = result['First name'] + ' ' + result['Last name']
    print(len(result), 'course participants parsed')
    return result


def parse_participants_cf_handles_spreadsheet(input_path: Path) -> dict[str, str]:
    result = pd.read_csv(input_path, sep=',', header=0, index_col=False,
                         usecols=['First name', 'Last name', 'CF Handle'])
    result['Fullname'] = result['First name'] + ' ' + result['Last name']
    return {res['Fullname']: res['CF Handle'] for _, res in result.iterrows()}


def generate_attempts_dataframe(cf_handles: list[str], problem_indexes: list[str],
                                submissions: dict[int, ShortSubmission]) -> pd.DataFrame:
    result = pd.DataFrame(index=cf_handles, columns=problem_indexes).infer_objects().fillna(0).astype(int)
    result.index.name = 'CF Handle'
    for submission in sorted(submissions.values(), key=lambda k: k.creation_time, reverse=True):
        cur_handle = submission.handle
        cur_problem = submission.problem_index

        if submission.verdict == 'OK' or submission.verdict == 'PARTIAL':
            if not result.loc[cur_handle, cur_problem]:
                # Consider only last result for each handle/problem pair
                result.loc[cur_handle, cur_problem] = submission.points
        
        # print(submission.handle, submission.problem_index, submission.verdict, submission.verdict == 'OK')
    result['Total'] = result.sum(axis=1) - (result['C'] if 'C' in ALL_PROBLEM_INDEXES else 0)
    return result


def clean_previous_results():
    tasks = 'ABCD'
    for t in tasks:
        path = f'data/JPlag/{t}'
        if os.path.exists(path):
            shutil.rmtree(path)
            
        path = f'data/JPlag/{t}_jp_results.jplag'
        if os.path.exists(path): os.remove(path)
    
    paths = (f'data/JPlag/jplag_results.csv', 
             f'data/{CONTEST_ID}_results.csv',
             f'data/{CONTEST_ID}_submits.json')
    for path in paths:        
        if os.path.exists(path): os.remove(path)
    

def main():
    clean_previous_results()
    contest_str = str(CONTEST_ID)
    status_path = DATA_DIR / (contest_str + '_submits.json')
    submissions_source_dir = contest_str
    final_results_path = DATA_DIR / (contest_str + '_results.csv')
    solutions_dir = contest_str + '_solutions'

    print(get_online_submission_count(CONTEST_ID), 'online submissions')
    download_contest_status(CONTEST_ID, status_path)
    submissions = load_contest_status(status_path)
    print('Used languages:',set([s['programming_language'] for s in submissions]))
    submissions = [s for s in submissions if s['programming_language'] in ACCEPTED_LANGUAGES]
    print('After filtering by language:', len(submissions))
    
    all_problem_indexes = get_contest_problem_indexes(CONTEST_ID)
    print('List of problems:', *all_problem_indexes)

    raw_submissions = parse_submissions(submissions)
    submissions_dict = filter_submissions(raw_submissions,
                                          LATE_SUBMISSION_IDS,
                                          exclude_handles=INSTRUCTOR_HANDLES,
                                          # submission_types=['CONTESTANT'],
                                          # submission_verdicts=['OK'],
                                          # problem_indexes="BCDEFGHIJKLMNOPQRSTUVWXYZ",
                                          finish_time=END_DATE
                                          )

    print(f'{len(submissions_dict)} submissions left after filtering')
    ordered_user_submissions = collect_submissions_per_user(submissions_dict)
    print(len(ordered_user_submissions), 'codeforces handles found')
    
    cf_handles = parse_participants_cf_handles_spreadsheet(
        CF_HANDLES_SPREADSHEET_FILE)
    participants_df = parse_course_participants(
        MOODLE_COURSE_PARTICIPANTS_FILE)

    print(f'{len(cf_handles)} CF handles parsed. Unique', len(set(cf_handles)))
    unknown_cf_handles = list(set(ordered_user_submissions.keys()) - set(cf_handles.values()))
    print('The following list of', len(unknown_cf_handles),'handles are missing:', *unknown_cf_handles)
    
    def generate_student_problem_solved_scores():
        attempts_df = generate_attempts_dataframe(list(ordered_user_submissions.keys()),
                                                  problem_indexes=ALL_PROBLEM_INDEXES.split(),
                                                  submissions=submissions_dict)
        cf_handle_df = pd.DataFrame(cf_handles.items(), columns=['Fullname', 'CF Handle'])
        final_result_df = (participants_df.merge(cf_handle_df, on='Fullname', how='left').fillna('')
                           .merge(attempts_df, left_on='CF Handle', right_index=True, how='left'))
        final_result_df.insert(
            final_result_df.columns.get_loc('CF Handle')+1, '-', '')
        
        print(final_result_df)
        final_result_df.drop('Fullname', axis=1).to_csv(
            final_results_path, index=False, encoding='utf-8-sig')

    # Grade Report Generation
    generate_student_problem_solved_scores()

    # Solution analysis and plagiarism detection.
    rename_submission_file_extensions()
    user_last_ok_submission_per_task = collect_last_user_submission_per_task(ordered_user_submissions, submissions_dict)

    # JPlag Similarity Detection
    jplag_measure_similarity(user_last_ok_submission_per_task, cf_handles, participants_df)
    calculate_jplag_metrics_and_statistics()


if __name__ == '__main__':
    main()
