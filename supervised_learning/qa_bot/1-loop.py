#!/usr/bin/env python3
"""
asdsadasdsa
"""

def main():
    """asdasdasdasdas"""
    exit_commands = {'exit', 'quit', 'goodbye', 'bye'}

    while True:
        question = input("Q: ").strip()
        if question.lower() in exit_commands:
            print("A: Goodbye")
            break
        print("A:")


if __name__ == "__main__":
    main()
