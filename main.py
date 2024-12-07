from randomSampling import runRandSamp

print("This is the group recreation of adversarial attacks on graphs\nTo begin, enter 1 to run a random sampling attack, 2 to run a genetic algorithm attack, 3 to run a reinforcement learning attack, or 4 to run a struct2vec attack.")
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
   print("Please put your gen algo here")