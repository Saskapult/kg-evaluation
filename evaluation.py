import json
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import dspy
import time
from kg_gen import KGGen
import argparse


def load_graph_from_json(file_path):
	with open(file_path, "r") as f:
		data = json.load(f)

	G = nx.DiGraph()

	for entity in data["entities"]:
		G.add_node(entity)

	for relation in data["relations"]:
		source, rel, target = relation
		G.add_edge(source, target, relation=rel)

	return G


def generate_embeddings(graph, model):
	node_embeddings = {node: model.encode(node).tolist() for node in graph.nodes}
	relation_embeddings = {
		rel: model.encode(rel).tolist() 
		for rel in set(edge[2]["relation"] 
		for edge in graph.edges(data=True))
	}
	return node_embeddings, relation_embeddings


# Retrieve top-k relevant nodes
def retrieve_relevant_nodes(query, node_embeddings, model, k=8):
	query_embedding = model.encode(query).reshape(1, -1)
	similarities = [
		(node, cosine_similarity(query_embedding, np.array(embed).reshape(1, -1))[0][0]) 
		for node, embed in node_embeddings.items()
	]
	similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
	return similarities[:k]


# Retrieve context from relationships
def retrieve_context(node, graph, depth=2):
	context = set()
	def explore_neighbors(current_node, current_depth):
		if current_depth > depth:
			return
		for neighbor in graph.neighbors(current_node):
			rel = graph[current_node][neighbor]["relation"]
			context.add(f"{current_node} {rel} {neighbor}.")
			explore_neighbors(neighbor, current_depth + 1)
	explore_neighbors(node, 1)
	return list(context)


# Use dspy to evaluate if the correct answer is in the context
def dspy_evaluate_response(correct_answer, context, judge_model):
	class EvalSignature(dspy.Signature):
		"""
		Determine whether the context contains the information stated in the correct answer. 
		Respond with True if yes, and False if no.
		"""
		correct_answer: int = dspy.InputField()
		context: str = dspy.InputField()
		contains_correct_information: bool = dspy.OutputField()
	evalualte = dspy.Predict(EvalSignature)

	with dspy.context(lm=dspy.LM(judge_model)):
		return evalualte(correct_answer=correct_answer, context=context).contains_correct_information


# Generate grpahs for a model and save them to disk
def generate_graphs(model, skip_existing=True, limit=None):
	kg = KGGen(model=model)
	graphs_dir = f"graphs/{model.replace("/", "_")}"
	os.makedirs(graphs_dir, exist_ok=True)
	inputs = [e for e in os.scandir("inputs") if e.is_file()]

	if limit:
		inputs = inputs[:limit]

	for i, f in enumerate(inputs):
		input_path = f"inputs/{f.name}"
		output_path = f"{graphs_dir}/{f.name}"
		print(f"Generate kg for {f.name} ({i+1}/{len(inputs)})")

		# Skip if already generated
		if skip_existing and os.path.isfile(output_path):
			print("Skip due to checkpoint detection")
			continue
		with open(input_path, "r") as f:
			data = json.load(f)
		st = time.time()
		graph = kg.generate(input_data=data["essay"])
		en = time.time()
		print(f"Graph generated in {(en-st):.2f} seconds")

		graph_data = {
			"entities": list(graph.entities),
			"edges": list(graph.edges),
			"relations": list(graph.relations),
		}
		# print(graph_data)
		with open(output_path, "w") as f:
			json.dump(graph_data, f, indent=2)
	
	return graphs_dir


# Evaluate accuracy
def evaluate_accuracy(questions_answers, node_embeddings, model, graph, output_file, judge_model):
	correct = 0
	results = []

	for correct_answer in questions_answers:
		top_nodes = retrieve_relevant_nodes(correct_answer, node_embeddings, model)
		context = []
		for node, _ in top_nodes:
			context.extend(retrieve_context(node, graph))
		context_text = " ".join(context)

		evaluation = dspy_evaluate_response(correct_answer, context_text, judge_model)
		results.append({
			"correct_answer": correct_answer,
			"retrieved_context": context_text,
			"evaluation": evaluation
		})
		correct += int(evaluation)

	accuracy = correct / len(questions_answers)
	results = {
		"results": results,
		"accuracy": accuracy,
	}

	# Save results to file
	with open(output_file, "w") as f:
		json.dump(results, f, indent=2)
	print(f"Results saved to {output_file}")


# For each file in the graphs directory, generate an evaluation
def generate_results(graphs_dir, judge_model):
	embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

	results_dir = f"results/{graphs_dir.split("/")[-1]}/{judge_model.replace("/", "_")}"
	os.makedirs(results_dir, exist_ok=True)

	graphs = [e.name for e in os.scandir(graphs_dir) if e.is_file()]
	for i, filename in enumerate(graphs):
		print(f"Evaluate {filename} ({i+1}/{len(graphs)})")
		
		input_file = f"inputs/{filename}"
		if not os.path.isfile(input_file):
			print(f"Could not find input file for {filename}, skipping")
			continue
		with open(input_file) as f:
			input_data = json.load(f)

		graph_file = f"{graphs_dir}/{filename}"
		G = load_graph_from_json(graph_file)

		print("Generate embeddings")
		st = time.time()
		node_embeddings, _ = generate_embeddings(G, embedding_model)
		en = time.time()
		print(f"Embeddings generated in {(en-st):.2f} seconds")

		print("Evaluate")
		output_file = f"{results_dir}/{filename}"
		evaluate_accuracy(input_data["answers"], node_embeddings, embedding_model, G, output_file, judge_model)

	return results_dir


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("model")
	parser.add_argument("judge", nargs="?")
	# parser.add_argument("adapter", nargs="?")
	parser.add_argument("--limit", type=int)
	args = parser.parse_args()

	# if args.adapter:
	# 	print(f"Using two-step adapter with {args.adapter}")
	# 	print("This is probably a bad idea!")
	# 	dspy.configure(adapter=dspy.TwoStepAdapter(dspy.LM(args.adapter)))

	d = generate_graphs(args.model, limit=args.limit)
	if args.judge:
		generate_results(d, args.judge)
	else:
		print("You will need to specify a judge LLM to get results")


if __name__ == "__main__":
	main()
