import jiwer
import jieba
def evaluate_model(ground_truths,predictions):
    # 将每个句子进行分词并重新组合成字符串
    ground_truths_words = [' '.join(jieba.lcut(sentence)) for sentence in ground_truths]
    predictions_words = [' '.join(jieba.lcut(sentence)) for sentence in predictions]
    
    # 计算词错误率（WER）
    wer = jiwer.wer(ground_truths_words, predictions_words)
    print(f"词错误率 (WER): {wer:.4f}")

    # 将分词后的句子重新组合成字符串，以便计算字符错误率（CER）
    ground_truth_str = ' '.join(ground_truths_words)
    prediction_str = ' '.join(predictions_words)

    # 计算字符错误率（CER）
    cer = jiwer.cer(ground_truth_str, prediction_str)
    print(f"字符错误率 (CER): {cer:.4f}")