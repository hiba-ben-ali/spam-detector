import joblib
from utils import nettoyer

model= joblib.load("spam_model.joblib")


print("Spam detector ready. Type a massage (or 'quit').")
while True:
    msg = input("\nMessage: ")
    if msg.lower() == "quit":
        break
    prob = model.predict_proba([nettoyer(msg)])[0][1]
    label= "SPAM" if prob >= 0.5 else "HAM (normal)"
    print(f"Result: {label}  (spam probability = {prob: .2%})")
