def load_suits(filepath):
    train_suits = []
    i = 0
    for Suit in ['Spades', 'Diamonds', 'Clubs', 'Hearts']:

        suit_path = os.path.join(filepath, Suit)

        images = [img for img in os.listdir(suit_path) if img.endswith('.jpg')]
        for img_name in images:
            train_suits.append(Train_suits())
            train_suits[i].name = Suit
            #image_path = os.path.join(suit_path, img_name)
            #train_suits[i].img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            train_suits[i].img = cv2.imread(os.path.join(suit_path, img_name), cv2.IMREAD_GRAYSCALE)
            i += 1
    return train_suits


best_rank_match_diff = 10000
best_suit_match_diff = 10000
best_rank_match_name = "Unknown"
best_suit_match_name = "Unknown"
i = 0


for Tsuit in train_suits:
                
                if qCard.suit_img.shape != Tsuit.img.shape:
                    qCard.suit_img = cv2.resize(qCard.suit_img, (Tsuit.img.shape[1], Tsuit.img.shape[0]))
                diff_img = cv2.absdiff(qCard.suit_img, Tsuit.img)
                suit_diff = int(np.sum(diff_img)/255)
                
                if suit_diff < best_suit_match_diff:
                    best_suit_diff_img = diff_img
                    best_suit_match_diff = suit_diff
                    best_suit_name = Tsuit.name 