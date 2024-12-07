from randomSampling import runRandSamp
from pyimports import Planetoid, NormalizeFeatures
from Genetic_Functions_gnn import Genetic_Algo
from Run_Evals import run_model, basic_run
from Struct2Vec import run_struct

Citseer = Planetoid(root='data/Planetoid', name='Citeseer', transform=NormalizeFeatures())
Cora = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
PubMed = Planetoid(root='data/Planetoid', name='PubMed', transform=NormalizeFeatures())

print("This is the group recreation of adversarial attacks on graphs\nTo begin, enter 1 to run a random sampling attack, 2 to run a genetic algorithm attack, 3 to run the GCN model without attacks. 4 to run the struc2vec model without attack")
while True:
  try:
    userIn = int(input("Enter 1, 2, 3, or 4: "))
    if userIn in range(1,5):
      break
    else:
      print("Invalid input")
  except ValueError:
    print("Invalid input")

if userIn == 1:
    print("Enter how many nodes you want to attack.")
    while True:
        try:
            userIn = int(input("Enter a number greater than 0: "))
            if userIn > 0:
                break
            else:
             print("Invalid input")
        except ValueError:
            print("Invalid input")
    runRandSamp(userIn)

elif userIn == 2:
  dataset = Citseer
  data = dataset[0]
  print("How many members in a population?: ")
  while True:
      try:
          userIn = int(input("Enter a number greater than 0: "))
          if userIn > 0:
              break
          else:
            print("Invalid input")
      except ValueError:
          print("Invalid input")  

  print("How many iterations should the GA run?: ")
  while True:
      try:
          userIn2 = int(input("Enter a number greater than 0: "))
          if userIn2 > 0:
              break
          else:
            print("Invalid input")
      except ValueError:
          print("Invalid input")

  Genetic_Algo(graph=data, dataset= dataset, n_population=userIn, iterations=userIn2)

elif userIn == 3:
  dataset = Citseer
  data = dataset[0]
  print("How many epochs for the GCN?: ")
  while True:
      try:
          userIn = int(input("Enter a number greater than 0: "))
          if userIn > 0:
              break
          else:
            print("Invalid input")
      except ValueError:
          print("Invalid input")  

  acc = basic_run(dataset,userIn)
  print(acc)
  
elif userIn == 4:
  dataset = Citseer
  data = dataset[0]
  print("What is the learning rate for the Struct2Vec?: ")
  while True:
      try:
          userIn = int(input("Enter a number less than or equal to 0.1: "))
          if userIn <= 0.1:
              break
          else:
            print("Invalid input")
      except ValueError:
          print("Invalid input")  

  print("How many hidden channels in the Struct2Vec?: ")
  while True:
      try:
          userIn2 = int(input("Enter 16, 32, 64, or 128:"))
          if userIn2 not in (16, 32, 64, 128):
              break
          else:
            print("Invalid input")
      except ValueError:
          print("Invalid input")  
  run_struct(userIn, userIn2)