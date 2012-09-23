package core;
import java.io.IOException;

import weka.core.converters.*;
import weka.core.*;
import preprocess.*;
import core.CommonMethords;
public class DataBaseTool {
	private String username;
	private String password;
	private DatabaseLoader dl;
	private DatabaseSaverEx ds;
	public void init(String uname,String pw) throws Exception{
		dl=new DatabaseLoader();
		ds=new DatabaseSaverEx();
		dl.setPassword(pw);
		dl.setUser(uname);
		ds.setUser(uname);
		ds.setPassword(pw);
		dl.connectToDatabase();
		ds.connectToDatabase();
	}
	public void insertTable(Instances data,String table) throws Exception{
		//在原有基础上进行添加操作
		ds.setTableName(table);
		ds.writeInstances(data);
	}
	public void insertTable(Instance data,String table) throws Exception{
		ds.setTableName(table);
		ds.writeInstance(data);
	}
	public Instances getTable(String tablename) throws Exception{
		Instances result;
		dl.setQuery("Select * from ".concat(tablename));
		result=dl.getDataSet();
		if(result==null) return dl.getDataSet();
		return result;
	}
	public Instances getTableStructure(String tablename) throws Exception{
		Instances result;
		dl.setQuery("Select * from ".concat(tablename));
		result=dl.getStructure();
		return result;
	}
	public void closeConnection(){
		ds=null;
		dl=null;
	}
}
