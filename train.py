"""
Nextgen AI - Training Script
Trains and saves the neural network.
"""

import os
import sys
import io
import argparse

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from brain import ChatBot

def main():
    parser = argparse.ArgumentParser(description='Nextgen AI model egitimi')
    parser.add_argument('--epochs', type=int, default=3000,
                        help='epoch sayisi (varsayilan 3000)')
    parser.add_argument('--no-demo', action='store_true',
                        help='egitim sonrasi test sohbetini atla (hizli CI icin)')
    args = parser.parse_args()

    print("=" * 50)
    print("  NEXTGEN AI - MODEL TRAINING")
    print("  Built from Scratch AI")
    print("=" * 50)
    print()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    intents_file = os.path.join(script_dir, 'intents.json')
    model_dir = os.path.join(script_dir, 'model')

    if not os.path.exists(intents_file):
        print(f"ERROR: {intents_file} not found!")
        sys.exit(1)

    bot = ChatBot()

    print("Training parameters:")
    print(f"  - Epochs: {args.epochs}")
    print("  - Learning Rate: 0.01")
    print()
    losses = bot.train_model(intents_file, epochs=args.epochs, learning_rate=0.01)

    print()
    print("=" * 50)

    bot.save_model(model_dir)

    print()
    print("Training completed!")
    print(f"Model saved: {model_dir}")
    print()

    if args.no_demo:
        return

    print("Start the web interface:")
    print("  python app.py")
    print()

    print("=" * 50)
    print("  TEST CHAT")
    print("=" * 50)
    print()

    test_messages = [
        "merhaba",
        "sen kimsin",
        "ne yapiyorsun",
        "tesekkurler",
        "gorusuruz",
        "bilim nedir",
        "yapay zeka hakkinda bilgi ver",
        "futbol nasil oynanir",
        "felsefe ne demek",
        "pizza nasil yapilir",
        "kedi hakkinda bilgi ver"
    ]

    for msg in test_messages:
        response = bot.get_response(msg)
        print(f"Sen: {msg}")
        print(f"AI:  {response}")
        print()


if __name__ == '__main__':
    main()
