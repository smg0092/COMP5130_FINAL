from pyimports import *

def run_model(data, dataset, epochs):

    class GCN(torch.nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()
            # torch.manual_seed(1234567)
            self.conv1 = GCNConv(dataset.num_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return x

    model = GCN(hidden_channels=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # Example: Train the model (simplified)
    for epoch in range(1, epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Perform a forward pass
        out = model(data.x, data.edge_index)
        predictions = out.argmax(dim=1)  # Get predicted classes

    # Calculate accuracy on the test set
    correct = (predictions[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())

    accuracy, precision, recall, f1 = evaluate_model(model, data)
    return acc, accuracy, precision, recall, f1

def run_plots(dataset):
    import matplotlib.pyplot as plt
    data = dataset[0]  # Get the first graph object.

    rows = []
    runs = []
    accuracies = []
    number_of_edges_removed = []
    num = 1000
    for i in range(10):  # Example loop for 5 iterations
        member = Member(data)
        member.remove_edges(i * num)

        fitness, accuracy = evaluate_member_fitness(member, dataset, 101)
        accuracies.append(accuracy)
        print(i)

        number_of_edges_removed.append(i*num)
        runs.append(i+1)

        row = [f"Run {i}", i*num, accuracy]
        rows.append(row)

    # Table headers
    headers = ["Run", "Edges Deleted", "Accuracy"]

    # Create the figure and axis
    fig, ax = plt.subplots()

    # Hide the default axes
    ax.axis('tight')
    ax.axis('off')

    # Create the table
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')

    # Style the table (optional)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(headers))))

    plt.savefig("table.png", bbox_inches="tight")
    # Show the table
    plt.show()

    #BOTH OF THEM SEPERTATE
    # Example data
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))  # 2 rows, 1 column

    # Plot the first graph
    axes[0].plot(runs, accuracies, color="blue", marker='o')
    axes[0].set_title("Line Graph 1")
    axes[0].set_xlabel("X-Axis")
    axes[0].set_ylabel("accuraties")

    # Plot the second graph
    axes[1].plot(runs, number_of_edges_removed,  color="red", marker='s')
    axes[1].set_title("Line Graph 2")
    axes[1].set_xlabel("X-Axis")
    axes[1].set_ylabel("number of edges removed")

    plt.tight_layout()
    plt.savefig("Plot.png")
    plt.show()

    # Plot the graph
    plt.plot(number_of_edges_removed, accuracies, label="accurcy with number_of_edges_removed", color="blue", marker='o')

    # Add title and labels
    plt.title("Line Graph")
    plt.xlabel("accuracies")
    plt.ylabel("number_of_edges_removed")

    # Add legend
    plt.legend()
    plt.savefig("bothComparison.png")
    plt.show()

def evaluate_model(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        predictions = out.argmax(dim=1)  # Predicted class labels
        true_labels = data.y[data.test_mask]  # True labels for the test set
        pred_labels = predictions[data.test_mask]  # Predictions for the test set

    # Convert to NumPy for sklearn metrics
    true_labels = true_labels.cpu().numpy()
    pred_labels = pred_labels.cpu().numpy()

    # Compute metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')

    # Print results
    return accuracy, precision, recall, f1

def basic_run(dataset):
    data = dataset[0]  # Get the first graph object.
    acc = run_model(data, dataset, 101)

    return acc

def evaluation_run(dataset):
    data = dataset[0]
    member = Member(data)
    fitness, accuracy = evaluate_member_fitness(member, dataset, 101)
    
    return accuracy

def print_dataset(data, dataset):
    print('======================')
    print(f'Dataset: {dataset}')
    print(f'Data type:')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print('======================')

    print(len(data))
    print("Data type of data graph object: ")
    print(type(data))

def average(a_list):
    avg = sum(a_list)/len(a_list)
    return avg

def avg_run_test():
    acc_list, accuracy_list, precision_list, recall_list, f1_list, final_avg = [],[],[],[],[],[]
    dataset_list = [Citseer, Cora, PubMed]
    dataset_names = ['Citseer', 'Cora', 'PubMed']

    with open(f"Averages.txt", "w") as file:
        for i in range(3):
            for j in range(30):
                acc1, accuracy, precision, recall, f1 = basic_run(dataset_list[i])
                acc_list.append(acc1), accuracy_list.append(accuracy), precision_list.append(precision), recall_list.append(recall), f1_list.append(f1)

            dat_acc_avg, dat_accuracy, dat_precision, dat_recall, dat_f1  = average(acc_list), average(accuracy_list), average(precision_list), average(recall_list), average(f1_list)
            file.write(f"30 Runs for {dataset_names[i]} Dataset:")
            file.write(f"{dataset_names[i]} avgerage acc: {dat_acc_avg}\n ")
            file.write(f"{dataset_names[i]} average accuracy score: {dat_accuracy}\n")
            file.write(f"{dataset_names[i]} average precision score: {dat_precision}\n ")
            file.write(f"{dataset_names[i]} average recall score: {dat_recall}\n ")
            file.write(f"{dataset_names[i]} average f1-score: {dat_f1}\n ")
            file.flush()

    