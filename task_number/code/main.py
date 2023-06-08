import sys

from hackathon_code.task_1 import run_task_1

def main():
    if len(sys.argv) != 3:
        return

    input1 = sys.argv[1]
    input2 = sys.argv[2]

    run_task_1(input1)

    # load trained model

    # export predictions

if __name__ == "__main__":
    main()
