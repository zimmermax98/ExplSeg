import os
import shutil
import csv
import numpy as np

explanations_dir = "L-CRP"
evaluation_dir = f"{os.path.basename(dir)}_evaluation"
n_samples = 50

correct_dir = os.path.join(explanations_dir, "explanations_correct")
incorrect_dir = os.path.join(explanations_dir, "explanations_incorrect")

correct_files = os.listdir(correct_dir)
incorrect_files = os.listdir(incorrect_dir)


evaluation_base_dir = evaluation_dir
i = 0
while os.path.exists(os.path.join(evaluation_dir)):
    i += 1
    evaluation_dir = f"{evaluation_base_dir}{i}"

print(evaluation_dir)
os.mkdir(evaluation_dir)

with open(os.path.join(evaluation_dir, "info.csv"), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Sample", "Name", "Correct/Incorrect"])
    n_correct = 0
    n_incorrect = 0

    for i_sample in range(n_samples):

        choice = np.random.choice(["correct", "incorrect"])

        if choice == "correct":
            file_index = np.random.randint(0, len(correct_files)-1)
            file_name = correct_files.pop(file_index)
            n_correct += 1
            if file_name in correct_files:
                file_index = incorrect_files.index(file_name)
                incorrect_files.pop(file_index)
        elif choice == "incorrect":
            file_index = np.random.randint(0, len(incorrect_files)-1)
            file_name = incorrect_files.pop(file_index)
            n_incorrect += 1
            if file_name in correct_files:
                file_index = correct_files.index(file_name)
                correct_files.pop(file_index)

        writer.writerow([f"{i_sample+1:2d}", file_name, choice])
        if choice == "correct":
            shutil.copy(os.path.join(correct_dir, file_name), os.path.join(evaluation_dir, f"{i_sample+1}.png"))
        elif choice == "incorrect":
            shutil.copy(os.path.join(incorrect_dir, file_name), os.path.join(evaluation_dir, f"{i_sample+1}.png"))


print(f"Total: {n_correct} correct, {n_incorrect} incorrect")
        