package Test;
import java.io.*;

import core.CommonMethords;
import core.DBScan;
import core.EXPLSA;
import weka.core.converters.*;
import weka.filters.*;
import preprocess.*;
import weka.clusterers.ClusterEvaluation;
import weka.core.*;
public class TestPreprocess {
	public static void main(String[] args) throws Exception{
		//新文档的分词处理,处理结果存入content_segmented等待进一步处理
		/*Segmenter seg=new Segmenter("content", "content_segmented" );
		seg.segment();
		//处理dir目录下的文档生成arff文档datafiltered。arff和dataRaw.arff
		//new TextToArffConverter().runsegmenterIntoArff(".//content_segmented//dir//");
		Dictionary dic = new Dictionary();
		Instances data = new TextToArffConverter().convertFileToTextARFFWithName("content_segmented");
		new CommonMethords().saveInstances(data, "testdata.arff");
		*/
		//以上为分词的步骤
		//以下为构造字典的过程
		Instances data = new Instances(new FileReader(new File(("testfiltered.arff"))));
		//Instances datafiltered=dic.constructDictionary(data);
		CommonMethords cm = new CommonMethords();
		//double[][] arrdata = cm.InstanceToArrays(data);
		
		int numIns = data.numInstances();
		int numad = (int)Math.ceil(numIns*0.2);
		int numWeb = numIns - numad;
		
		// 第一个属性为ID 不记做属性
		EXPLSA plsa = new EXPLSA(data, data.numInstances(), 15, data.numAttributes()-1);
		//PLSA plsa = new PLSA(arrdata, numad, numWeb, 15, data.numAttributes());
		plsa.PLSA_TEMStep();
		
		//Instances datatopic = cm.ArrasysToInstance(plsa.Pz_d_out);
		Instances datatopic = cm.addIDInfo(data, plsa.Pz_d);
		cm.saveInstances(datatopic, "datatopic.arff");

	}
}
