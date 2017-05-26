import unicodecsv
from datetime import datetime as dt
import numpy as np
from collections import defaultdict

def read_csv(filename):
    with open(filename, 'rb') as f:
        reader = unicodecsv.DictReader(f)
        return list(reader)


def get_unique_set(sets, key):
    unique_set = set()
    for row in sets:
        unique_set.add(row[key])
    return unique_set


def change_key(sets, okey, nkey):
    for row in sets:
        row[nkey] = row[okey]
        del [row[okey]]


def parse_date(date):
    if date == '':
        return None
    else:
        return dt.strptime(date, '%Y-%m-%d')


def parse_maybe_int(i):
    if i == '':
        return None
    else:
        return int(i)


enrollments = read_csv('/home/loveshadev/PycharmProjects/Udacity/TestProject/DataAnalysis/enrollments.csv')
daily_engagement = read_csv('/home/loveshadev/PycharmProjects/Udacity/TestProject/DataAnalysis/daily_engagement.csv')
project_submissions = read_csv(
    '/home/loveshadev/PycharmProjects/Udacity/TestProject/DataAnalysis/project_submissions.csv')

### For each of these three tables, find the number of rows in the table and
### the number of unique students in the table. To find the number of unique
### students, you might want to create a set of the account keys in each table.
for enrollment in enrollments:
    enrollment['cancel_date'] = parse_date(enrollment['cancel_date'])
    enrollment['days_to_cancel'] = parse_maybe_int(enrollment['days_to_cancel'])
    enrollment['is_canceled'] = enrollment['is_canceled'] == 'True'
    enrollment['is_udacity'] = enrollment['is_udacity'] == 'True'
    enrollment['join_date'] = parse_date(enrollment['join_date'])

for engagement_record in daily_engagement:
    engagement_record['lessons_completed'] = int(float(engagement_record['lessons_completed']))
    engagement_record['num_courses_visited'] = int(float(engagement_record['num_courses_visited']))
    engagement_record['projects_completed'] = int(float(engagement_record['projects_completed']))
    engagement_record['total_minutes_visited'] = float(engagement_record['total_minutes_visited'])
    engagement_record['utc_date'] = parse_date(engagement_record['utc_date'])

for submission in project_submissions:
    submission['completion_date'] = parse_date(submission['completion_date'])
    submission['creation_date'] = parse_date(submission['creation_date'])

enrollment_num_rows = len(enrollments)
unique_set_enrollment = get_unique_set(enrollments, 'account_key')
enrollment_num_unique_students = len(unique_set_enrollment)

engagement_num_rows = len(daily_engagement)
unique_set_engagement = get_unique_set(daily_engagement, 'acct')
engagement_num_unique_students = len(unique_set_engagement)

submission_num_rows = len(project_submissions)
unique_set_submission = get_unique_set(project_submissions, 'account_key')
submission_num_unique_students = len(unique_set_submission)

change_key(daily_engagement, 'acct', 'account_key')

for enrollment in enrollments:
    student = enrollment['account_key']
    if student not in unique_set_engagement:
        print enrollment
        break

num_problem_students = 0
for enrollment in enrollments:
    student = enrollment['account_key']
    if (student not in unique_set_engagement and
            enrollment['join_date'] != enrollment['cancel_date']):
        num_problem_students += 1

udacity_test_accounts = set()
for enrollment in enrollments:
    if enrollment['is_udacity']:
        udacity_test_accounts.add(enrollment['account_key'])


def remove_udacity_accounts(data):
    non_udacity_data = []
    for data_point in data:
        if data_point['account_key'] not in udacity_test_accounts:
            non_udacity_data.append(data_point)
    return non_udacity_data

non_udacity_enrollments = remove_udacity_accounts(enrollments)
non_udacity_engagement = remove_udacity_accounts(daily_engagement)
non_udacity_submissions = remove_udacity_accounts(project_submissions)

paid_students = {}

for enrollment in non_udacity_enrollments:
    if (not enrollment['is_canceled'] or
            enrollment['days_to_cancel'] > 7):
        account_key = enrollment['account_key']
        enrollment_date = enrollment['join_date']
        if (account_key not in paid_students or
                enrollment_date > paid_students[account_key]):
            paid_students[account_key] = enrollment_date

def within_one_week(join_date, engagement_date):
    time_delta = engagement_date-join_date
    return time_delta.days < 7

def remove_free_trial_cancels(data):
    new_data = []
    for data_point in data:
        if data_point['account_key'] in paid_students:
            new_data.append(data_point)
    return new_data

paid_enrollments = remove_free_trial_cancels(non_udacity_enrollments)
paid_engagement = remove_free_trial_cancels(non_udacity_engagement)
paid_submissions = remove_free_trial_cancels(non_udacity_submissions)


paid_engagement_in_first_week = []
for engagement_record in paid_engagement:
    account_key = engagement_record['account_key']
    join_date = paid_students[account_key]
    engagement_record_date = engagement_record['utc_date']
    if within_one_week(join_date, engagement_record_date):
        paid_engagement_in_first_week.append(engagement_record)

#print len(paid_engagement_in_first_week)

def group_data(data, key_name):
    grouped_data = defaultdict(list)
    for data_point in data:
        key = data_point[key_name]
        grouped_data[key].append(data_point)
    return grouped_data

engagement_by_account = group_data(paid_engagement_in_first_week, 'account_key')

def sum_grouped_items(grouped_data, field_name):
    summed_data = {}
    for key, data_points in grouped_data.items():
        total = 0
        for data_point in data_points:
            total += data_point[field_name]
        summed_data[key] = total
    return summed_data

total_minutes_by_account = sum_grouped_items(engagement_by_account,
                                             'total_minutes_visited')

#print total_minutes_by_account
lessons_completed_by_account = sum_grouped_items(engagement_by_account,'lessons_completed')
def print_describe(datas):
    print "Mean : "+str(np.mean(datas))
    print "Standard Deviation : "+str(np.std(datas))
    print "Minimum : "+str(np.min(datas))
    print "Maximum : "+str(np.max(datas))

print_describe(lessons_completed_by_account.values())

for engagement_record in paid_engagement:
    if engagement_record['num_courses_visited'] > 0:
        engagement_record['has_visited'] = 1
    else:
        engagement_record['has_visited'] = 0

days_visited_by_account = sum_grouped_items(engagement_by_account,'has_visited')
print_describe(days_visited_by_account.values())