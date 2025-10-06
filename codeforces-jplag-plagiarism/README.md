1. Create .env file and fill it similarly as in .env.example file
1. Download answers with handles from a test in moodle and unzip them in 'cf' folder
    - **cf** folder should look like:
        - Ivan Ivanov_123...
            - onlinetext.html (the cf handle lies inside)
        - John Jhonov_456...
            - onlinetext.html
1. Run script 'parse_handles_from_moodle_test.py'. It produces 'data/cf_handle.csv' file (check this file on correctness)
1. Download submissions from CF and put under folder f'data/{contest_id}'
1. Download list of moodle students and put them into 'data/course_participants.csv' file. Columns: 'First name,Last name,Email address'
1. Put `jplag-6.2.0.jar` file in 'data/JPlag' folder.
1. Change constants specific to you in codeforces.py file and run it.
    - Usually need to change: `CONTEST_ID` and `END_DATE`
    - f'{contest_id}_results.csv' with points for each cf problem considering last submission
    - It will produce jplag_results.csv with pairwise submission matches
