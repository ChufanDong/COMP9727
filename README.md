2025.7.28 HenryBai
完成了对文本和上线时间的处理，可以直接运行tests里的test_text.py进行调试结果查看

(1) 对文本和上线时间的处理方法在./preprocessing/text_preprocess.py中

(2)文本处理为text_process(text: str) -> str,对文本进行：
小写化，分词，只保留单词和连词（state-of-the-art），lemmatize，去除无意义词。
最终返回用空格连接的结果("a, b 10 c."->"a b c")

(3)上线时间的处理为transform_quarter(date: str) -> str，对接收的animes里的"aired"列进行上线时间和完结时间的提取，并处理为年份-季度(01,02,03,04)的形式。
时间为?或Not available的项为NULL，只有年份的项默认季度为01
返回结果例("2018-Q1","2018-Q2")

(3)对数据集的处理方法在./preprocessing/preprocess_pipline.py, 直接调用final_preprocess(anime, profile, review)传入3个数据集即可得到处理后的数据。
目前只实现了对animes里描述列（'synopsis'）和日期列（'aired'）的处理。