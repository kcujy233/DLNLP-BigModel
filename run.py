from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer
from transformers import BloomTokenizerFast, BloomForCausalLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
'''ChatYuan-large-v1'''
tokenizer = T5Tokenizer.from_pretrained("ClueAI/ChatYuan-large-v1")
model = T5ForConditionalGeneration.from_pretrained("ClueAI/ChatYuan-large-v1")
device = torch.device('cuda')
model.to(device)
def preprocess(text):
  text = text.replace("\n", "\\n").replace("\t", "\\t")
  return text
def postprocess(text):
  return text.replace("\\n", "\n").replace("\\t", "\t")
def answer(text, sample=True, top_p=1, temperature=0.7):
  '''sample：是否抽样。生成任务，可以设置为True;
  top_p：0-1之间，生成的内容越多样'''
  text = preprocess(text)
  encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=768, return_tensors="pt").to(device)
  if not sample:
    out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=512, num_beams=1, length_penalty=0.6)
  else:
    out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_new_tokens=512, do_sample=True, top_p=top_p, temperature=temperature, no_repeat_ngram_size=3)
  out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
  return postprocess(out_text[0])
input_text0 = "帮我写一个请假条，我因为新冠不舒服，需要请假3天，请领导批准"
input_text1 = "“这款手机的电池续航非常出色。屏幕显示效果也很清晰，运行速度非常快。总体来说，我非常满意这次购物。”分析一下上述语句的感情色彩"
input_text2 = "“酒店房间设施陈旧，床垫很硬，让人难以入睡。前台服务态度冷淡，与客人沟通不耐烦。这次的住宿体验令人失望。”分析一下上述语句的感情色彩"
input_text3 = "“这部电影的剧情设定很有创意，视觉效果震撼。虽然部分角色塑造略显薄弱，但总体而言还是值得一看的佳作。”分析一下上述语句的感情色彩"
input_text4 = "写一个诗歌，关于冬天"
input_text5 = "“自然语言处理是人工智能领域的一个重要分支，它关注计算机理解、解释和生成人类自然语言的方法。NLP旨在使计算机能够与人类有效地交流，处理大量文本数据，提取出有意义的信息，从而辅助决策、回答问题等。NLP技术已广泛应用于众多领域，如搜索引擎、聊天机器人、机器翻译、文本摘要、情感分析等。”将上述语句翻译成英文"
input_text6 = "根据今天发布的官方报告，本月初中国东北部的洪涝灾害已造成至少50人死亡，23人失踪。受灾地区包括辽宁、吉林和黑龙江三个省份。数以千计的房屋被毁，约34万人被迫撤离家园。为抗击洪水，当局动用了大量的救援物资和人力资源，同时也呼吁民众提供支持。政府承诺将投入更多资金重建受灾地区，加快修复基础设施，恢复正常生产和生活秩序。总结以上内容的摘要："
input_list = [input_text0, input_text1, input_text2, input_text3, input_text4, input_text5, input_text6]
for i, input_text in enumerate(input_list):
  input_text = "用户：" + input_text + "\n小元："
  print(f"示例{i}".center(50, "="))
  output_text = answer(input_text)
  print(f"{input_text}{output_text}")
print("end...")

'''bloom-1b4-zh'''
tokenizer = BloomTokenizerFast.from_pretrained('Langboat/bloom-1b4-zh')
model = BloomForCausalLM.from_pretrained('Langboat/bloom-1b4-zh').cuda()
input = tokenizer.encode('根据今天发布的官方报告，本月初中国东北部的洪涝灾害已造成至少50人死亡，23人失踪。受灾地区包括辽宁、吉林和黑龙江三个省份。数以千计的房屋被毁，约34万人被迫撤离家园。为抗击洪水，当局动用了大量的救援物资和人力资源，同时也呼吁民众提供支持。政府承诺将投入更多资金重建受灾地区，加快修复基础设施。总结以上内容的摘要：', return_tensors='pt').cuda()
print(tokenizer.batch_decode(model.generate(input, max_new_tokens=500)))

'''GPT2-3.5B-chinese'''
tokenizer = GPT2Tokenizer.from_pretrained('IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese')
model = GPT2LMHeadModel.from_pretrained('IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese')
input_txt = "根据今天发布的官方报告，本月初中国东北部的洪涝灾害已造成至少50人死亡，23人失踪。受灾地区包括辽宁、吉林和黑龙江三个省份。数以千计的房屋被毁，约34万人被迫撤离家园。为抗击洪水，当局动用了大量的救援物资和人力资源，同时也呼吁民众提供支持。政府承诺将投入更多资金重建受灾地区，加快修复基础设施，恢复正常生产和生活秩序。总结以上内容的摘要："

n_steps = 300  # 进行8步解码
input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"]
output = model.generate(input_ids, max_new_tokens=n_steps, do_sample=False)
print(tokenizer.decode(output[0]))
