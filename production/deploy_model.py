from azureml.core import Workspace, Run

def fetch_job_details(workspace, job_name):
    # Retrieve the run (job) by name
    runs = Run.list(workspace)
    run = next((r for r in runs if r.id.endswith(job_name)), None)

    if not run:
        raise Exception(f"No run found with name: {job_name}")

    # Fetch and print inputs
    print("Inputs:")
    for input_name, data_asset, asset_uri in fetch_inputs(run):
        print(f"Input name: {input_name}")
        print(f"Data asset: {data_asset}")
        print(f"Asset URI: {asset_uri}\n")

    # Fetch and print outputs
    print("Outputs:")
    for output_name, model_name, asset_uri in fetch_outputs(run):
        print(f"Output name: {output_name}")
        print(f"Model: {model_name}")
        print(f"Asset URI: {asset_uri}\n")

    # Fetch and print tags
    print("Tags:")
    for key, value in run.get_tags().items():
        print(f"{key}: {value}")

    # Fetch and print parameters
    print("Params:")
    for key, value in run.get_details()['runDefinition']['arguments']:
        print(f"{key}: {value}")

    # Fetch and print metrics
    print("Metrics:")
    for metric_name, metric_value in run.get_metrics().items():
        print(f"{metric_name}:\n{metric_value}\n")

def fetch_inputs(run):
    inputs = run.get_details()['runDefinition']['inputDatasets']
    for input_name, details in inputs.items():
        data_asset = details['dataset'].split(':')[1]
        asset_uri = details['dataLocation']['uri']
        yield input_name, data_asset, asset_uri

def fetch_outputs(run):
    outputs = run.get_details()['runDefinition']['outputDatasets']
    for output_name, details in outputs.items():
        model_name = details['model']['id'].split(':')[1]
        asset_uri = details['dataLocation']['uri']
        yield output_name, model_name, asset_uri

if __name__ == "__main__":
    # Replace with your workspace details
    subscription_id = 'bf0717bf-dfd1-4019-a2b6-aa46e3899a4d'
    resource_group = 'assignment-snobin'
    workspace_name = 'assignmentsnobin'

    # Load the workspace
    ws = Workspace(subscription_id, resource_group, workspace_name)

    # Replace 'your_job_name' with the actual job name
    job_name = 'gray_store_x3ltwv1lb1'

    # Fetch job details
    fetch_job_details(ws, job_name)


# from azureml.core import Workspace, Experiment, Run
# import os 

# # Replace with your workspace details
# subscription_id = 'bf0717bf-dfd1-4019-a2b6-aa46e3899a4d'
# resource_group = 'assignment-snobin'
# workspace_name = 'assignmentsnobin'

# # Replace 'your_experiment_name' and 'your_run_id' with actual experiment and run details
# experiment_name = 'coursework-ml-compute-human-action-classification'
# run_id = 'gray_store_x3ltwv1lb1'



# # Load the workspace
# ws = Workspace(subscription_id, resource_group, workspace_name)

# # Load the experiment
# exp = Experiment(ws, experiment_name)

# run = next(exp.get_runs())
# files = run.get_file_names()
# print(files)


