package core;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.core.Instances;

public class JustForTest {

	/**
	 * @param args
	 * @throws IOException 
	 * @throws FileNotFoundException 
	 */
	public static void main(String[] args) throws FileNotFoundException, IOException {
		// TODO Auto-generated method stub
		Instances data = new Instances(new FileReader(new File("datafiltered.arff")));
		//for(int i = 0 ; i < data.numInstances(); i++)
			System.out.println(data.instance(0));
		for (int i = 0; i <data.instance(0).numValues(); i++){
			System.out.println(data.instance(0).valueSparse(i));
		System.out.println(data.instance(0).index(i)+"\n");
		}
		
		System.out.println(data.instance(0));
		for (int i = 0; i < data.instance(0).numValues(); i++){
			//System.out.println(data.instance(0).index(i));
			System.out.println(data.instance(0).value(i));
		}
	}

}
