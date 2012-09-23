package Test;
import java.io.*;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import core.*;
import weka.core.converters.*;
import CircleOfTheProcess.*;
public class Test {
	public static void main(String[] args) throws Exception{
		DataBaseTool dbt = new DataBaseTool();
		dbt.init("root","4432519");
		dbt.insertTable(new Instances(new FileReader(new File("aaa.arff"))), "IDmatchName");
		/*DatabaseSaver ds= new DatabaseSaver();
		ds.setPassword("4432519");
		ds.setUser("root");
		ds.connectToDatabase();
		Instances data=new Instances(new FileReader(new File("dataRaw.arff")));
		data.setRelationName("IDmatchName");
		ds.setInstances(data);
		ds.writeBatch();*/
	}
}
