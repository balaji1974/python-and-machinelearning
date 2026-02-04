import random
number1 = random.randint(1, 100)
number2 = random.randint(1, 100)

answer = input("What is " + str(number1) + " + " + str(number2) + "? ")

if(int(answer) == number1 + number2):
    print("Correct!")
else:
    print("Incorrect. The answer is " + str(number1 + number2) + ".")
    print("You can try again if you want.")