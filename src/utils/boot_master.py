import os
import sys
import boto3


def manage_master(action):
    # 1. Retrieve the region from environment variables (same logic as the Client)
    region = os.getenv("AWS_REGION", "us-east-1")

    try:
        # Initialize Boto3 clients
        ssm = boto3.client('ssm', region_name=region)
        asg = boto3.client('autoscaling', region_name=region)

        # 2. Retrieve the Master ASG name from SSM
        param_name = "/drf/ec2/master_asg_name"
        print(f" [INIT] Reading Master ASG name from SSM ({param_name})...")
        response = ssm.get_parameter(Name=param_name)
        master_asg_name = response['Parameter']['Value']

        # 3. Calculate desired capacity based on the command
        desired_capacity = 1 if action == "start" else 0
        min_size = 0 if action == "stop" else 1

        print(f" [AWS] Setting ASG '{master_asg_name}' to DesiredCapacity={desired_capacity}...")

        # 4. Update the Auto Scaling Group
        asg.update_auto_scaling_group(
            AutoScalingGroupName=master_asg_name,
            MinSize=min_size,
            DesiredCapacity=desired_capacity,
            MaxSize=1
        )

        state = "STARTED 🟢" if action == "start" else "STOPPED 🔴"
        print(f"\n {'=' * 40}")
        print(f" SUCCESS! The Master node has been {state}.")
        if action == "start":
            print(" Please wait 1-2 minutes for the instance to boot and start Docker.")
        print(f" {'=' * 40}\n")

    except ssm.exceptions.ParameterNotFound:
        print(f"\n [CRITICAL ERROR] Parameter '{param_name}' not found on AWS SSM!")
        print(" You must create it on AWS Parameter Store setting the Master ASG name as 'Value'.")
        sys.exit(1)
    except Exception as e:
        print(f"\n [ERROR] Unable to communicate with AWS: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Simple CLI to accept "start" or "stop" as an argument
    if len(sys.argv) < 2 or sys.argv[1].lower() not in ["start", "stop"]:
        print("\n [USAGE] Run the script by passing 'start' or 'stop':")
        print("   python3 manage_master.py start")
        print("   python3 manage_master.py stop\n")
        sys.exit(1)

    command = sys.argv[1].lower()
    manage_master(command)