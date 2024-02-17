import cv2
import numpy as np
import time
import os
import Cards
import video

IM_WIDTH = 1280
IM_HEIGHT = 720
FRAME_RATE = 30

human_card = []  # untuk manusia
bot_card = []    # untuk bot
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

                # posisi kartu dalam frame
                card_pos_x = card.center[0]  # Posisi X 
                midpoint = image.shape[1] // 2  # Titik tengah lebar gambar
                
                if rank != "Unknown" and suit != "Unknown":
                    card_identified = f"{rank} of {suit}"
                    current_time = time.time()
                    if (current_time - last_card_time) > time_threshold:
                        if card_pos_x < midpoint:  # sisi kiri
                            human_card.append(card_identified)
                        else:  # sisi kanan
                            bot_card.append(card_identified)
                        last_card_time = current_time
                    #image = Cards.draw_results(image, card)

    # Display
    ypos_human = 350
    ypos_bot = 350
    for card_name in human_card:
        cv2.putText(image, card_name, (10, ypos_human), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        ypos_human += 30
    for card_name in bot_card:
        cv2.putText(image, card_name, (IM_WIDTH - 200, ypos_bot), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        ypos_bot += 30

    cv2.imshow("Card Detector", image)
    #cv2.imshow("pre", pre_proc)

    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cam_quit = 1

cv2.destroyAllWindows()
videostream.stop()