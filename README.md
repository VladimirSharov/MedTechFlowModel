# Surgery Room Simulation Project

## Purpose
This project simulates the operations of a hospital surgery facility, focusing on the efficiency and resource utilization of preparation rooms, operation theatres, and recovery rooms. The simulation aims to optimize patient flow and minimize operational bottlenecks in a hospital setting.

## How It Works
The simulation uses a discrete-event simulation model built with the SimPy library. It simulates patient flow through various stages: preparation, operation, and recovery. The simulation runs multiple scenarios with different configurations of preparation and recovery rooms to analyze their impact on overall efficiency.

## Key Components
1. **Patient Arrival Process**: Patients arrive at the hospital following an exponential distribution, mimicking the randomness of real-world patient arrivals.
2. **Treatment Stages**: The simulation includes distinct stages for patient preparation, operation, and recovery, each with its own process and resource allocation.
3. **Resource Allocation**: The number of preparation, operation, and recovery units are adjustable parameters in the simulation.
4. **Statistical Analysis**: The simulation employs batch means methods and other statistical techniques to provide reliable estimates of system performance.

## Adjustable Parameters
- `NUMBER_PREPARATION_UNITS`, `NUMBER_OPERATION_UNITS`, `NUMBER_RECOVERY_UNITS`: Define the capacity of each type of unit.
- `PATIENT_ARRIVAL_MEAN`, `PATIENT_PREPARATION_MEAN`, `PATIENT_OPERATION_MEAN`, `PATIENT_RECOVERY_MEAN`: Control the mean times for various stages.
- `SIM_TIME`: Determines the total simulation time.
- `DEBUG`: Enable or disable debug mode for additional output.

## Running the Simulation
1. Install dependencies from `requirements.txt`.
2. Execute the simulation script.
3. The simulation will run predefined scenarios with different configurations (e.g., 3 preparation rooms and 4 recovery rooms).
4. Analyze the output data for key metrics like queue lengths, idle capacities, and operation blocking probabilities.

## Output and Analysis
- The simulation generates data on the average queue length, idle time of units, and the probability of operations being blocked due to full recovery units.
- Statistical analyses such as calculating confidence intervals and standard deviations are performed.
- Output is visualized in tables and graphs for easier interpretation.

## Customization
Users can modify the parameters in the script to test different scenarios or hypotheses about hospital operations.
