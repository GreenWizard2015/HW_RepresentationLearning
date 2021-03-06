# Representation Learning

Репозиторий для задач с занятия 19 и 20. Сиамские сети и автокодировщики весьма похожи, поэтому я решил их объединить в один репозиторий, что упростит их сравнение.

## План работы

- [x] Реализовать сиамскую сеть.
  - [x] Оценить качество модели при различных значениях параметра `alpha` функции `triplet_loss`.
  - [x] Оценить точность и confusion matrix стандартного способа классификации сиамской сетью.
  - [x] Оценить точность и confusion matrix классификации алгоритмом KMeans на основе представления выученного сиамской сетью.

- [x] Реализовать автокодировщик.
  - [x] Реализовать VAE.
  - [x] Оценить точность и confusion matrix классификации алгоритмом KMeans на основе представления выученного автокодировщиком.
  - [x] Применить автокодировщик для детекции аномалий.
  - [x] Применить автокодировщик для восстановления повреждённых участков изображения.

- [ ] (Опционально) Попробовать совместить сиамскую сеть с автокодировщиком, чтоб получить более качественное представление.
