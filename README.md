# WarCard-OpenCv-Game
==
Fitur Game
--
Game warcard yang menggunakan fitur deteksi kamera OpenCV
Untuk Deteksi menggunakan metode template matching yang ada pada OpenCV
Untuk kamera menggunakan Ip camera pada android
Musuh pada permainan ini adalah komputer yang sudah diprogram

Alur permainan adalah:
--
1. Melakukan scan kartu untuk Player (manusia) yang nantinya akan masuk ke dalam list kartu player
2. Melakukan scan kartu musuh (bot) yang nantinya akan masuk ke dalam list kartu bot
3. ketika sudah memiliki kartu permainan dapat dimulai
4. player harus meletakan kartu ke area deteksi kamera, jika kartu yang diletakkan ada pada list kartu maka kartu akan valid sebaliknya jika tidak ada maka kartu tidak ada terdeteksi
5. setelah kartu player sudah terdeteksi, bot akan mengeluarkan kartu dari list secara otomatis
6. selanjutnya program game akan membandingkan nilai dari kartu tersebut
7. nilai yang lebih besar akan menang
8. apabila nilai dari kartu yang dikeluarkan sama maka sistem akan menghitung nilai total kartu yang ada pada list kartu player dan bot
9. nilai total kartu tersebut akan dibandingkan
10. nilai yang lebih besar akan menang
