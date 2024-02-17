import cv2
import numpy as np
import time
import os
import random
# Impor modul Cards dan video Anda di sini, jika ada
import Cards
import video

IM_WIDTH = 1280
IM_HEIGHT = 720
FRAME_RATE = 30

human_card = []  # untuk manusia
bot_card = []    # untuk bot
game_active = False
collect_for = None  # 'human' atau 'bot'
last_card_time = time.time()  # terakhir ditambahkan
time_threshold = 5  # Jeda waktu
frame_rate_calc = 1
freq = cv2.getTickFrequency()

font = cv2.FONT_HERSHEY_SIMPLEX

videostream = video.VideoStream((IM_WIDTH, IM_HEIGHT), FRAME_RATE, 1).start()
time.sleep(1)

path = os.path.dirname(os.path.abspath(__file__))
train_ranks = Cards.load_ranks(path + '/Card_Imgs/')
train_suits = Cards.load_suits(path + '/Card_Imgs/')

# Mendefinisikan nilai kartu
values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'Jack': 11, 'Queen': 12, 'King': 13, 'Ace': 14}

def play_round(human_card, bot_card):
    if not human_card or not bot_card:
        return "No cards to play", "", ""

    # Memilih kartu secara acak dan menghapus dari daftar
    human_card_played = random.choice(human_card)
    bot_card_played = random.choice(bot_card)
    human_card.remove(human_card_played)
    bot_card.remove(bot_card_played)

    human_card_played_split = human_card_played.split(' of ')
    bot_card_played_split = bot_card_played.split(' of ')

    human_card_value = values.get(human_card_played_split[0], 0)
    bot_card_value = values.get(bot_card_played_split[0], 0)

    if human_card_value > bot_card_value:
        return "Human wins", human_card_played_split, bot_card_played_split
    elif human_card_value < bot_card_value:
        return "Bot wins", human_card_played_split, bot_card_played_split
    else:
        return "It's a tie", human_card_played_split, bot_card_played_split

cam_quit = 0

while cam_quit == 0:
    image = videostream.read()
    t1 = cv2.getTickCount()

    pre_proc = Cards.preprocess_image(image)
    cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)

    if len(cnts_sort) != 0:
        for i in range(len(cnts_sort)):
            if cnt_is_card[i] == 1:
                card = Cards.preprocess_card(cnts_sort[i], image)
                rank, suit, rank_diff, suit_diff = Cards.match_card(card, train_ranks, train_suits)

                card_pos_x = card.center[0]
                midpoint = image.shape[1] // 2
                
                if rank != "Unknown" and suit != "Unknown":
                    card_identified = f"{rank} of {suit}"
                    current_time = time.time()
                    if (current_time - last_card_time) > time_threshold:
                        if collect_for == 'human' and card_pos_x < midpoint:
                            human_card.append(card_identified)
                        elif collect_for == 'bot' and card_pos_x >= midpoint:
                            bot_card.append(card_identified)
                        last_card_time = current_time

    # Display
    ypos_human = 350
    ypos_bot = 350
    for card_name in human_card:
        cv2.putText(image, card_name, (10, ypos_human), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        ypos_human += 30
    for card_name in bot_card:
        cv2.putText(image, card_name, (IM_WIDTH - 200, ypos_bot), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        ypos_bot += 30

    if game_active and len(human_card) > 0 and len(bot_card) > 0:
        round_result, human_card_played, bot_card_played = play_round(human_card, bot_card)
        cv2.putText(image, f"Round result: {round_result}", (IM_WIDTH // 2 - 100, 50), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"Human played: {' '.join(human_card_played)}", (IM_WIDTH // 2 - 100, 80), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f"Bot played: {' '.join(bot_card_played)}", (IM_WIDTH // 2 - 100, 110), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Card Detector", image)
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    key = cv2.waitKey(1) & 0xFF
    if key == ord("1"):
        collect_for = 'human'
    elif key == ord("2"):
        collect_for = 'bot'
    elif key == ord("n") or key == ord("N"):
        game_active = not game_active
    elif key == ord("r") or key == ord("R"):
        human_card.clear()
        bot_card.clear()
        game_active = False
        collect_for = None
    elif key == ord("q"):
        cam_quit = 1

cv2.destroyAllWindows()
videostream.stop()
