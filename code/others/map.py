import random

def generate_map(size=10, coin_count=20):
    # Initialize a blank map
    game_map = [['.' for _ in range(size)] for _ in range(size)]

    # Place coins randomly on the map
    coins_placed = 0
    while coins_placed < coin_count:
        x = random.randint(0, size - 1)
        y = random.randint(0, size - 1)
        if game_map[x][y] == '.':
            game_map[x][y] = 'C'
            coins_placed += 1

    return game_map

def print_map(game_map):
    for row in game_map:
        print(' '.join(row))

if __name__ == "__main__":
    # Generate and display the map
    map_size = 10
    coins = 20  # Number of coins
    generated_map = generate_map(size=map_size, coin_count=coins)
    print_map(generated_map)
