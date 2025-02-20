import random

n = random.randint(1,99)
guess = int(input("Enter a number between 1 to 99: "))

while n != guess:
    print()
    if n < guess:
        print("Guess is high")
    elif n > guess:
        print("Guess is low")
    else:
        break
    guess = int(input("Enter a number between 1 to 99: "))
    print()
print ("You guessed it right")



