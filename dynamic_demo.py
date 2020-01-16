import web
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
import urllib
import sys
import json
import tensorflow.keras as keras 

# 获取隐向量并归一化
model = keras.models.load_model('model_d30_interrupted.model')
X = model.layers[2].embeddings.numpy()
normalized_X = [x / np.sqrt(x.dot(x)) for x in X]

# 建立zid到index的映射
anchor_list = pd.read_csv('anchor_list.csv')
anchor_list = anchor_list.reset_index()
anchor_list['index'] = anchor_list['index'] + 1
zid2index = {zid: index for zid, index in zip(anchor_list['live_uid'], anchor_list['index'])}
index2zid = {index: zid for index, zid in zip(anchor_list['index'], anchor_list['live_uid'])}

# 训练最近邻模型
nbrs = NearestNeighbors(n_neighbors=10, n_jobs=-1).fit(normalized_X)

def query_by_url(url):
    res_data = urllib.request.urlopen(url, timeout=30)
    res = res_data.read()
    res_json = json.loads(res)
    return res_json

def query_info_by_ids(ids):
    info_dic = {}
    per = 150
    for i in range(len(ids) // per + 1):
        try:
            info_url = "http://apigateway.inke.srv/user/infos?id=%s" % ",".join(ids[i * per: (i + 1) * per])
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
        query_point = normalized_X[zid2index[zid]]
        distances, indices = nbrs.kneighbors([query_point])

        similar_anchor_scores = [(2 - dist**2) / 2 for dist in distances[0]]
        similar_anchor_indices = indices[0]
        similar_anchor_zid = [index2zid[idx] for idx in similar_anchor_indices]
        info_dic = query_info_by_ids(similar_anchor_zid)

        # 写网页
        html_str = "<!DOCTYPE html><html>" \
           "<head><meta http-equiv="'"Content-Type"'" content="'"text/html;charset=utf-8"'">" \
           "<title>基于item2vec的相似主播</title>" \
           "</head>" \
           "<body>" \
           "<hr><table border=""∂1"">" \
           "<tr bgcolor=""#C0C0C0"">" \
           "<th>zid</th>" \
           "<th>头像</th>" \
           "<th>直播间</th>" \
           "<th>余项相似度</th>" \
           "</tr>"

        for zid, score in zip(similar_anchor_zid, similar_anchor_scores):
            detail = info_dic.get(zid)
            html_str += "<tr align=""center""  bgcolor=""#FFFFF0"">"
            html_str += '<td><div style="width:180px;word-wrap:break-word;" >%s</td>' % str(zid) # 添加uid
            html_str += "<td><img src=""%s"" height=""150"" width=""150""></td>" % detail.get("portrait","").decode('utf-8') # 添加头像
            html_str += '<td><a href="http://www.inke.com/live.html?uid=%s">VIEW LIVE</a></td>' % zid # 添加直播间链接
            html_str += '<td><div style="width:180px;word-wrap:break-word;" >%s</td>' % str(score) # 添加相似度
            html_str += "</tr>"
        html_str += "</table></body></html>"
        return html_str

    except KeyError:
        return '该主播不在训练集中'

class index:
    def GET(self, any):
        param = web.input(zid=None)
        return build_html(param.zid)

urls = ('/(.*)', 'index')

if __name__=='__main__':
    app = web.application(urls, globals())
    app.run()

