import plotly.graph_objects as go
import simpy
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import expon
import os
from datetime import datetime

# initialization module
NUMBER_PREPARATION_UNITS = 3
NUMBER_OPERATION_UNITS = 1
NUMBER_RECOVERY_UNITS = 3

PATIENT_ARRIVAL_MEAN = 3

PATIENT_PREPARATION_MEAN = 6
PATIENT_PREPARATION_STD = 0.5

PATIENT_OPERATION_MEAN = 2.0
PATIENT_OPERATION_STD = 0.5

PATIENT_RECOVERY_MEAN = 3.0
PATIENT_RECOVERY_STD = 0.5

normal_times = [
    PATIENT_PREPARATION_MEAN,
    PATIENT_OPERATION_STD,
    PATIENT_OPERATION_MEAN,
    PATIENT_OPERATION_STD,
    PATIENT_RECOVERY_MEAN,
    PATIENT_RECOVERY_STD]

SIM_TIME = 100


def patient_arrival(env, pr, ot, rr):
    # IDs for patients
    next_patient_id = 0
    while True:

        # exponential distribution for arrivals
        next_patient_time = expon.rvs(scale=PATIENT_ARRIVAL_MEAN, size=1)
        # Wait for the patient
        yield env.timeout(next_patient_time)
        next_patient_id += 1
        print('%3d arrives at %.2f' % (next_patient_id, env.now))

        time_of_arrival = env.now

        # pass as parameters $pr, $ot, $rr which are the ressources,
        # and $normal_times, wich is the array containing the times. This parameter
        # can depend on the type of patient.
        env.process(prepare(env, pr, ot, rr, next_patient_id,
                    time_of_arrival, normal_times))


def prepare(env, pr, ot, rr, patient_number, time_of_arrival, process_times):
    with pr.request() as req:
        print('%3d enters the preparation queue at %.2f' %
              (patient_number, env.now))

        queue_in_preparation = env.now
        # array just for the arrival times
        arrivals_preparation.append(queue_in_preparation)
        length = len(pr.queue)
        tme_in_queue_preparation.append(queue_in_preparation)
        len_in_queue_preparation.append(length)

        yield req
        print('%3d starts preparation at %.2f' % (patient_number, env.now))

        queue_out_preparation = env.now
        length = len(pr.queue)
        tme_in_queue_preparation.append(queue_out_preparation)
        len_in_queue_preparation.append(length)

        # normal distribution for the preparation process
        r_normal = norm.rvs(loc=process_times[0],
                            scale=process_times[1], size=1)
        yield env.timeout(r_normal)
        print("%3d preparation duration was %.2f" % (patient_number, r_normal))

        departures_preparation.append(env.now)
        time_in_queue_preparation = queue_out_preparation - queue_in_preparation
        in_queue_preparation.append(time_in_queue_preparation)

        env.process(operate(env, ot, rr,
                    patient_number, time_of_arrival, process_times))


def operate(env, ot, rr, patient_number, time_of_arrival, process_times):
    with ot.request() as req:
        print('%3d enters the operation queue at %.2f' %
              (patient_number, env.now))

        queue_in_operation = env.now
        arrivals_operation.append(queue_in_operation)
        length = len(ot.queue)
        tme_in_queue_operation.append(queue_in_operation)
        len_in_queue_operation.append(length)

        yield req
        print('%3d starts operation at %.2f' %
              (patient_number, env.now))

        queue_out_operation = env.now
        length = len(ot.queue)
        tme_in_queue_operation.append(queue_out_operation)
        len_in_queue_operation.append(length)

        # normal distribution for the operating process
        r_normal = norm.rvs(loc=process_times[2],
                            scale=process_times[3], size=1)
        yield env.timeout(r_normal)
        print("%3d operation duration was %.2f" % (patient_number, r_normal))

        departures_operation.append(env.now)
        time_in_queue_operation = queue_out_operation - queue_in_operation
        in_queue_operation.append(time_in_queue_operation)

        env.process(recover(env, rr,
                    patient_number, time_of_arrival, process_times))


def recover(env, rr, patient_number, time_of_arrival, process_times):
    with rr.request() as req:
        print('%3d enters the recovery queue at %.2f' %
              (patient_number, env.now))

        queue_in_recovery = env.now
        arrivals_recovery.append(queue_in_recovery)
        length = len(rr.queue)
        tme_in_queue_recovery.append(queue_in_recovery)
        len_in_queue_recovery.append(length)

        yield req
        print('%3d leaves the recovery queue at %.2f' %
              (patient_number, env.now))

        queue_out_recovery = env.now
        length = len(rr.queue)
        tme_in_queue_recovery.append(queue_out_recovery)
        len_in_queue_recovery.append(length)

        # normal distribution for the recovery process
        r_normal = norm.rvs(loc=process_times[4],
                            scale=process_times[5], size=1)
        yield env.timeout(r_normal)
        print("%3d recovery duration was %.2f" % (patient_number, r_normal))

        time_of_departure = env.now
        departures_recovery.append(time_of_departure)
        time_in_system = time_of_departure - time_of_arrival
        in_system.append(time_in_system)
        time_in_queue_recovery = queue_out_recovery - queue_in_recovery
        in_queue_recovery.append(time_in_queue_recovery)


def avg_line(df_length):
    # finds the time average number of patients in the waiting line
    # use the next row to figure out how long the queue was
    df_length['delta_time'] = df_length['time'].shift(-1)-df_length['time']
    # drop the last row because it would have an infinite delta time
    df_length = df_length[0:-1]
    avg = np.average(df_length['len'], weights=df_length['delta_time'])
    return avg


def server_utilization(df_length):
    # finds the server utilization
    sum_server_free = df_length[df_length['len'] == 0]['delta_time'].sum()
    # the process begins with the server empty
    first_event = df_length['time'].iloc[0]
    sum_server_free = sum_server_free + first_event
    utilization = round((1 - sum_server_free / SIM_TIME) * 100, 2)
    return utilization


def not_allowed_perc(df_length, not_allowed_number):
    # finds the percentage of time of patients on queue not allowed to be waiting
    sum_not_allowed = df_length[df_length['len']
                                >= not_allowed_number]['delta_time'].sum()
    not_allowed = round((sum_not_allowed / SIM_TIME) * 100, 2)
    return not_allowed


def queue_analytics(time_in_queue, len_in_queue, in_queue):
    df_time = pd.DataFrame(time_in_queue, columns=['time'])
    df_len = pd.DataFrame(len_in_queue, columns=['len'])
    df_length = pd.concat([df_time, df_len], axis=1)
    avg_length = avg_line(df_length)
    utilization = server_utilization(df_length)
    #not_allowed = not_allowed_perc(df_length)
    avg_delay_inqueue = np.mean(in_queue)
    return (avg_length, utilization, avg_delay_inqueue, df_length)


if __name__ == '__main__':
    # arrays to keep track
    arrivals_preparation, departures_preparation = [], []
    in_queue_preparation, in_preparation = [], []
    tme_in_queue_preparation, len_in_queue_preparation = [], []

    arrivals_operation, departures_operation = [], []
    in_queue_operation, in_operation = [], []
    tme_in_queue_operation, len_in_queue_operation = [], []

    arrivals_recovery, departures_recovery = [], []
    in_queue_recovery, in_recovery = [], []
    tme_in_queue_recovery, len_in_queue_recovery = [], []

    in_system = []

    # set up the environment
    env = simpy.Environment()
    # defining resources
    pr = simpy.Resource(env, capacity=NUMBER_PREPARATION_UNITS)
    ot = simpy.Resource(env, capacity=NUMBER_OPERATION_UNITS)
    rr = simpy.Resource(env, capacity=NUMBER_OPERATION_UNITS)
    # selecting a random seed for the probability distributions
    RANDOM_SEEDS = [1234, 5678, 9012, 3456, 7890]
    np.random.seed(seed=RANDOM_SEEDS[2])

    # TODO : implement a way to pass the distribution we want to have
    #  inside the patients, so that we know wich patients carry wich process times. The idea is to have two
    #  types of patients, and a random distibution between the two.

    # defining the patient arrival process
    env.process(patient_arrival(env, pr, ot, rr))
    # run the simultion
    env.run(until=SIM_TIME)

    # Analyses part :

    (avg_length_prep,  utilization_prep, avg_delay_inqueue_preparation, df_length_prep) = queue_analytics(
        tme_in_queue_preparation, len_in_queue_preparation, in_queue_preparation)
    (avg_length_op,  utilization_op, avg_delay_inqueue_operation, df_length_op) = queue_analytics(
        tme_in_queue_operation, len_in_queue_operation, in_queue_operation)
    (avg_length_rec,  utilization_rec, avg_delay_inqueue_recovery, df_length_rec) = queue_analytics(
        tme_in_queue_recovery, len_in_queue_recovery, in_queue_recovery)

    df_arrival = pd.DataFrame(arrivals_preparation,   columns=['arrivals'])
    df_start_operation = pd.DataFrame(
        arrivals_operation,   columns=['arrivals_operation'])
    df_end_operation = pd.DataFrame(
        departures_operation,   columns=['departures_operation'])
    df_departures = pd.DataFrame(departures_recovery, columns=['departures'])
    df_chart = pd.concat([df_arrival, df_start_operation,
                          df_end_operation, df_departures], axis=1)

    # average time spent in the system
    avg_delay_insyst = np.mean(in_system)

    print('  ')
    print('The average delay in preparation queue is %.2f' %
          (avg_delay_inqueue_preparation))
    print('The average delay in operation queue is %.2f' %
          (avg_delay_inqueue_operation))
    print('The average delay in recovery queue is %.2f' %
          (avg_delay_inqueue_recovery))
    print('The average delay in system is %.2f' % (avg_delay_insyst))
    print('The average number of patients in preparation queue is %.2f' %
          (avg_length_prep))
    print('The average number of patients in operation queue is %.2f' %
          (avg_length_op))
    print('The average number of patients in recovery queue is %.2f' %
          (avg_length_rec))
    print('The utilization of the preparation server is %.2f' %
          (utilization_prep))
    print('The utilization of the operation server is %.2f' % (utilization_op))
    print('The utilization of the recovery server is %.2f' % (utilization_rec))


# Function to save figure to a specific folder with a timestamp
def save_figure(fig, filename_prefix):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    folder_path = 'figures'  # Change to your preferred folder name
    os.makedirs(folder_path, exist_ok=True)
    filename = f"{filename_prefix}_{timestamp}.html"
    file_path = os.path.join(folder_path, filename)
    fig.write_html(file_path, auto_open=True)

# Common style for all figures
layout_style = dict(
    width=0.7 * 1920,  # Adjust to your screen resolution
    height=0.7 * 1080,
    xaxis_title=dict(text='Time', standoff=10),
    yaxis_title=dict(text='Number of Patients', standoff=10),
    font=dict(color='white'),  # White font for better contrast
    paper_bgcolor='#1f2630',  # Dark background color
    plot_bgcolor='#1f2630',   # Dark plot background color
    margin=dict(l=50, r=50, b=50, t=80),  # Margin for better visibility
    showlegend=True,
    legend=dict(orientation='h', x=0.5, y=1.05),  # Centered legend
)

# Plotting the arrivals and departures from the different services
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_chart['arrivals'], mode='markers', name='Arrivals'))
fig.add_trace(go.Scatter(
    x=df_chart['arrivals_operation'], mode='markers', name='Arrivals Operation'))
fig.add_trace(go.Scatter(
    x=df_chart['departures_operation'], mode='markers', name='Departures Operation'))
fig.add_trace(go.Scatter(
    x=df_chart['departures'], mode='markers', name='Departures'))
fig.update_layout(title='Arrivals & Departures at the Operation Center', **layout_style)
save_figure(fig, 'arrivals_departures')

# Plotting the preparation queue
fig1 = go.Figure(go.Waterfall(x=df_length_prep['time'],
                              y=df_length_prep['len'],
                              measure=['absolute'] * 100,
                              connector={"line": {"color": "blue"}}))
fig1.update_layout(title='Number of Patients in Preparation Queue', **layout_style)
save_figure(fig1, 'preparation_queue')

# Plotting the operation queue
fig2 = go.Figure(go.Waterfall(x=df_length_op['time'],
                              y=df_length_op['len'],
                              measure=['absolute'] * 100,
                              connector={"line": {"color": "blue"}}))
fig2.update_layout(title='Number of Patients in Operation Queue', **layout_style)
save_figure(fig2, 'operation_queue')

# Plotting the recovery queue
fig3 = go.Figure(go.Waterfall(x=df_length_rec['time'],
                              y=df_length_rec['len'],
                              measure=['absolute'] * 100,
                              connector={"line": {"color": "blue"}}))
fig3.update_layout(title='Number of Patients in Recovery Queue', **layout_style)
save_figure(fig3, 'recovery_queue')
