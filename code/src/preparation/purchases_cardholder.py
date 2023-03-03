import sys
import preparation

if __name__ == "__main__":
    if len(sys.argv) == 1:
        preparation.purchases_cardholder()
        
        
    else:
        n = sys.argv[1]
        preparation.purchases_cardholder(n)