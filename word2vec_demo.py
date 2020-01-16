import web
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
import urllib
# import sys
import json
import tensorflow.keras as keras 

# reload(sys)
# sys.setdefaultencoding('utf-8')

# 获取隐向量并归一化
data = pd.read_csv('word.txt', sep=' ', header=None)
zids = data[0]
ebd = data.values[:, 1:31]
normalized_ebd = [x / np.sqrt(x.dot(x)) for x in ebd]

# 建立zid到index的映射
zid2index = {zid: index for index, zid in enumerate(zids)}
index2zid = {index: zid for index, zid in enumerate(zids)}

# 训练最近邻模型
nbrs = NearestNeighbors(n_neighbors=20, n_jobs=-1).fit(normalized_ebd)

def query_by_url(url):
	res_data = urllib.request.urlopen(url, timeout=30)
	res = res_data.read()
	res_json = json.loads(res)
    return res_json

def query_info_by_ids(ids):
    info_dic = {}
    for zid in ids:
        try:
            info_url = "http://apigateway.inke.srv/user/infos?id=%s" % zid
            info_list = query_by_url(info_url)
            if info_list["dm_error"] == 0 and len(info_list["users"]) > 0:
                for info in info_list["users"]:
                    info_dic[str(info["id"])] = info
        except Exception as e:
            print(e)
            continue
    return info_dic

def build_html(zid):
	# 查询输入uid对应的向量，若没有则报错
	try:
		zid = int(zid)
		query_point = normalized_ebd[zid2index[zid]]
		distances, indices = nbrs.kneighbors([query_point])

		similar_anchor_scores = [(2 - dist**2) / 2 for dist in distances[0]]
		similar_anchor_indices = indices[0]
		similar_anchor_zid = [index2zid[idx] for idx in similar_anchor_indices]
		info_dic = query_info_by_ids(similar_anchor_zid)

		# 写网页
	    html_str = "<!DOCTYPE html><html>" \
	       "<head><meta http-equiv="'"Content-Type"'" content="'"text/html;charset=utf-8"'">" \
	       "<title>Word2Vec By Jian Huoyong</title>" \
	       "</head>" \
	       "<body>" \
	       "<hr><table border=""∂1"">" \
	       "<tr bgcolor=""#C0C0C0"">" \
	       "<th>zid</th>" \
	       "<th>昵称</th>" \
	       "<th>头像</th>" \
	       "<th>直播间</th>" \
	       "<th>余项相似度</th>" \
	       "</tr>"

	    for zid, score in zip(similar_anchor_zid, similar_anchor_scores):
	    	detail = info_dic[str(zid)]
	    	html_str += "<tr align=""center""  bgcolor=""#FFFFF0"">"
	        html_str += '<td><div style="width:180px;word-wrap:break-word;" >%s</td>' % str(zid) # 添加uid
	        try:
                nick = detail['nick']
            except UnicodeDecodeError:
                nick = '\u9760\u0021\u0020\u4ec0\u4e48\u7834\u70c2\u6635\u79f0\uff01'
            html_str += "<td>%s</td>" % nick
	        html_str += "<td><img src=""%s"" height=""150"" width=""150""></td>" % detail["portrait"] # 添加头像
	        html_str += '<td><a href="http://www.inke.com/live.html?uid=%s">VIEW LIVE</a></td>' % zid # 添加直播间链接
	        html_str += '<td><div style="width:180px;word-wrap:break-word;" >%s</td>' % str(score) # 添加相似度
	        html_str += "</tr>"
	    html_str += "</table></body></html>"
	    return html_str
	except KeyError:
		return 'This zid is unavaiable.'

class index:
    def GET(self, zid):
        param = web.input(zid=None)
        return build_html(param.zid)

urls = ('/(.*)', 'index')

if __name__=='__main__':
    app = web.application(urls, globals())
    app.run()
