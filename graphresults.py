#!/usr/bin/python

import subprocess
import sys
import re
import matplotlib.pyplot as plt

# Prints usage for script
def usage(program):
	print 'Usage: {} league'.format(program)

    
### Run main execution ###

if __name__ == "__main__":

	# Exit if usage incorrect
	if len(sys.argv) != 2:
		usage(sys.argv[0])
		sys.exit(1)

	# Capture analysis script output
	output = subprocess.check_output(["./analyze.py", sys.argv[1]])
	output = re.split("[,:\n]+", output)

	# Compile values into lists
	keys = []
	accuracies = []
	precisions = []
	recalls = []
	f1 = []
	for i in range(0,25,5):
		keys.append(output[i])
		accuracies.append(float(output[i+1]))
		precisions.append(float(output[i+2]))
		recalls.append(float(output[i+3]))
		f1.append(float(output[i+4]))

	# Calculate averages
	acc_mean = sum(accuracies)/len(accuracies)
	pre_mean = sum(precisions)/len(precisions)
	recall_mean = sum(recalls)/len(recalls)
	f1_mean = sum(f1)/len(f1)
	acc_avg = []
	pre_avg = []
	recall_avg = []
	f1_avg = []
	for i in range(0,5):
		acc_avg.append(acc_mean)
		pre_avg.append(pre_mean)
		recall_avg.append(recall_mean)
		f1_avg.append(f1_mean)

	# Plotting
	fig, axs = plt.subplots(2,2)
	fig.set_figheight(8)
	fig.set_figwidth(10)

	# Accuracies
	axs[0,0].scatter(keys, accuracies, label='Accuracy', marker='o', color='b')
	axs[0,0].plot(keys, acc_avg, label='Average', linestyle='--', color='r')
	axs[0,0].set_title(sys.argv[1] + " Accuracy Statistics")
	axs[0,0].set_xlabel("Models")
	axs[0,0].set_ylabel("Accuracies")

	# Precisions
	axs[0,1].scatter(keys, precisions, label='Precision', marker='o', color='b')
	axs[0,1].plot(keys, pre_avg, label='Average', linestyle='--', color='r')
	axs[0,1].set_title(sys.argv[1] + " Precision Statistics")
	axs[0,1].set_xlabel("Models")
	axs[0,1].set_ylabel("Precisions")

	# Recalls
	axs[1,0].scatter(keys, recalls, label='Recall', marker='o', color='b')
	axs[1,0].plot(keys, recall_avg, label='Average', linestyle='--', color='r')
	axs[1,0].set_title(sys.argv[1] + " Recall Statistics")
	axs[1,0].set_xlabel("Models")
	axs[1,0].set_ylabel("Recalls")
	
	# F1
	axs[1,1].scatter(keys, f1, label='F1', marker='o', color='b')
	axs[1,1].plot(keys, f1_avg, label='Average', linestyle='--', color='r')
	axs[1,1].set_title(sys.argv[1] + " F1 Statistics")
	axs[1,1].set_xlabel("Models")
	axs[1,1].set_ylabel("F1 Score")
	
	# Show graphs and save figure
	plt.tight_layout(w_pad=1.5, h_pad=1.5)
	fig.savefig(sys.argv[1] + '_results/' + sys.argv[1] + '_eval_metrics.png')
	plt.show()
	plt.close()


