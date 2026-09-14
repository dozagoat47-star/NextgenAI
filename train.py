"""
Nextgen AI - Training Script
Trains and saves the neural network.
"""

import os
import sys
import io
import argparse

# Log yönlendirmeli çalışırken bile canlı (satır tamponlu) çıktı
if sys.stdout is not None:
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                      line_buffering=True)

from brain import ChatBot

def main():
    parser = argparse.ArgumentParser(description='Nextgen AI model egitimi')
    parser.add_argument('--epochs', type=int, default=500,
                        help='epoch sayisi (varsayilan 500)')
    parser.add_argument('--learning-rate', '--lr', dest='learning_rate', type=float,
                        default=0.001, help='baslangic ogrenme hizi (varsayilan 0.001)')
    parser.add_argument('--no-demo', action='store_true',
                        help='egitim sonrasi test sohbetini atla (hizli CI icin)')
    parser.add_argument('--all-intents', action='store_true',
                        help='tum intentleri siniflandiriciya ogret (varsayilan: '
                             'yalnizca sohbet intentleri; bilgiler retrieval ile)')
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
    print(f"  - Learning Rate: {args.learning_rate}")
    print()
    losses = bot.train_model(intents_file, epochs=args.epochs,
                             learning_rate=args.learning_rate,
                             conversational_only=not args.all_intents)

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
