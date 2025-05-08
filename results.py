import json 
import argparse
import os


def display_all(results_directory):
	scores = []
	for f in os.scandir(results_directory):
		with open(f"{results_directory}/{f.name}", "r") as ff:
			data = json.load(ff)
		print(f"{f.name} - {data["accuracy"]*100:.2f}%")
		scores.append(data["accuracy"])
	scores.sort()
	print(f"Median accuracy: {scores[len(scores)//2]*100:.2f}%")
	print(f"Mean accuracy: {sum(scores)/len(scores)*100:.2f}%")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("results_directory")
	args = parser.parse_args()

	display_all(args.results_directory)


if __name__ == "__main__":
	main()
