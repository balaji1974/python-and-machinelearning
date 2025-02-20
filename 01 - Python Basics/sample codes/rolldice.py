import random
roll_again='y'

while roll_again=='y':
    print("The dice rolled and value is.....")
    print(random.randint(1,7))
    roll_again = input("Do you want to roll again (y/n): ")
