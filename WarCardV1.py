import cv2
import numpy as np
import time
import os
# Impor modul Cards dan video Anda di sini, jika ada
import Cards
import video
import random


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
values = {'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5, 'Six': 6, 'Seven': 7, 'Eight': 8, 'Nine': 9, 'Ten': 10, 'Jack': 11, 'Queen': 12, 'King': 13, 'Ace': 14}
human_card_value = 0
bot_card_value = 0
card_text_h = "No cards in hand"
card_text_b = "No cards in hand"
hasil = "Belum ada"
font = cv2.FONT_HERSHEY_SIMPLEX

videostream = video.VideoStream((IM_WIDTH, IM_HEIGHT), FRAME_RATE, 1).start()
time.sleep(1)

path = os.path.dirname(os.path.abspath(__file__))
train_ranks = Cards.load_ranks(path + '/Card_Imgs/')
train_suits = Cards.load_suits(path + '/Card_Imgs/')


cam_quit = 0


def calculate_total_value(cards):
    total_value = 0
    for card in cards:
        rank = card.split(' of ')[0]
        total_value += values.get(rank, 0)
    return total_value

#def get_human_card_choice(human_card):
    print("Kartu yang tersedia: ")
    for card in human_card:
        print(card)
    choice = input("Pilih kartu Anda (ketikkan nama kartu): ")
    if choice in human_card:
        return choice
    else:
        print("Kartu tidak ditemukan. Silakan coba lagi.")
        return None


def get_human_card_choice(human_card):
    for idx, card in enumerate(human_card):
        print(f"{idx}: {card}")
    choice = input("Pilih kartu Anda (masukkan nomor): ")
    try:
        selected_card = human_card[int(choice)]
        return selected_card
    except (IndexError, ValueError):
        print("Pilihan tidak valid. Silakan coba lagi.")
        return None


#def get_bot_card_choice(bot_card):
    for idx, card in enumerate(bot_card):
        print(f"{idx}: {card}")
    choice = input("Pilih kartu Anda (masukkan nomor): ")
    try:
        selected_card = bot_card[int(choice)]
        return selected_card
    except (IndexError, ValueError):
        print("Pilihan tidak valid. Silakan coba lagi.")
        return None

while cam_quit == 0:
    image = videostream.read()
    t1 = cv2.getTickCount()

    pre_proc = Cards.preprocess_image(image)
    cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)


    # Display
    ypos_human = 350
    ypos_bot = 350
 
    total_human_value = calculate_total_value(human_card)
    total_bot_value = calculate_total_value(bot_card)

    

    key = cv2.waitKey(1) & 0xFF
    Game = 0
    if key == ord("1"):
        Game = 1
    if key == ord("2"):
        Game = 2
    if key == ord("3"):
        Game = 3
    elif key == ord("q"):
        cam_quit = 1
        
    if Game == 1:
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
                            human_card.append(card_identified)
        

    elif Game == 2:
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
                            bot_card.append(card_identified)
        
    elif Game == 3:
        if human_card and bot_card:
            selected_human_card = get_human_card_choice(human_card)
            if selected_human_card:
                human_card.remove(selected_human_card)
                card_text_h = f"Human Card: {selected_human_card}"
                human_card_value = values.get(selected_human_card.split(' of ')[0], 0)
            else:
                card_text_h = "No cards in hand"

            if bot_card:
                    random_bot_card = random.choice(bot_card)
                    bot_card.remove(random_bot_card)
                    card_text_b = f"Bot Card: {random_bot_card}"
                    bot_card_value = values.get(random_bot_card.split(' of ')[0], 0)        
            else:
                card_text_b = "No cards in hand"
        else:
            card_text_h = "Tidak ada kartu tersisa"
            card_text_b = "Tidak ada kartu tersisa"
    

    if human_card_value < bot_card_value:
        hasil = "Bot win"
    elif human_card_value > bot_card_value:
        hasil = "Player win"
    elif human_card_value == bot_card_value:
        hasil = "Draw"
        if total_human_value > total_bot_value:
            hasil = "Player win"
        elif total_bot_value > total_human_value:
            hasil = "Bot win"


    for card_name in human_card:
        cv2.putText(image, card_name, (10, ypos_human), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        ypos_human += 30
    for card_name in bot_card:
            cv2.putText(image, card_name, (IM_WIDTH - 250, ypos_bot), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            ypos_bot += 30


    

    cv2.putText(image, f"Human Random: {card_text_h}", (10, 50), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, f"Bot Random: {card_text_b}", (10, 75), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"Hasil: {hasil}", (10, 100), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"Total Human Value: {total_human_value}", (10, 330), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"Total Bot Value: {total_bot_value}", (IM_WIDTH - 250, 330), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Card Detector", image)
    print(human_card)
    print(bot_card)
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1


cv2.destroyAllWindows()
videostream.stop()