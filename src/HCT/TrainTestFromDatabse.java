package HCT;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.io.Serializable;
import java.net.UnknownHostException;
import java.util.StringTokenizer;
import java.util.Vector;

import com.mongodb.BasicDBObject;
import com.mongodb.DB;
import com.mongodb.DBCollection;
import com.mongodb.DBCursor;
import com.mongodb.DBObject;
import com.mongodb.MongoClient;
import com.mongodb.WriteResult;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_print_interface;
import libsvm.svm_problem;

public class TrainTestFromDatabse {

	public static void main(String[] args) {
		// TODO Auto-generated method stub

		String DataDir = "data/" ;
		
		String TrainFileName =DataDir + "train.txt";
		String ModelFileName = TrainFileName + ".model";
		String TestFileName = DataDir + "test.txt";
		String PredictOutputFile = DataDir + "res.txt";
		
		/** 1. 数据训练接断 **/
		
		
		try {
			
			svm_parameter param = SetSVMParameters(2,2); //g,c
			svm_problem  prob = CreateProblemFromFile(TrainFileName);
			
			//Insert Record to MongoDB
			String IP = "127.0.0.1";
			int port = 27017;
			String dbName = "PersonProfile";
			
			InsertProblemToDB(prob,IP,port);
			prob = ReadProblemFromDB(IP,port,dbName);
			
			//double accu = CrossValidation(prob, param, 3); // problem, parameter, fold
			
			//使用Grid Search 寻找参数 C and Gamma
			param = GridSearchParameter(prob, param); //取得較佳的參數
			
			//System.out.println("Cross Validation Accuracy Result: " + accu + " %");
			
			svm_model model = svm.svm_train(prob,param);  		// 训练资料成为 model 
			//svm.svm_save_model(ModelFileName,model);  //一般作法，储存Support Vector等资讯至档案。
			
			//将模型以Base64 String 的形式存入档案
			StoreModeltoFile(model,ModelFileName); 
			System.out.println("完成数据训练:" + ModelFileName);
			
			
			
		} catch (IOException e) {
		
			e.printStackTrace();
		}
		
		/** 2. 测试训练模型结果 **/
		
		try {
			

			
			//svm_model model = svm.svm_load_model(argv[i+1]);
			svm_model PredictModel  = ReadModelFromFile(ModelFileName);
			predict(TestFileName,PredictOutputFile,PredictModel);
					
			
		} catch (ClassNotFoundException | IOException e) {
			
			e.printStackTrace();
		}
		
		
	}
	
	private static svm_parameter GridSearchParameter(svm_problem prob, svm_parameter param) {
		
		param.C = 3;
		param.gamma = 3;
		
		int c_array[] = {-8,-5,-1, 0, 2, 4, 8, 10}; 		
		int g_array[] = {-8,-5,-1, 0, 2, 4, 8, 10}; 		
		
		double TopAccu = 0.0 ;
		double BetterC = 3;
		double BetterG = 3;
		
		//寻找预测正确率最高之参数
		for(double cc : c_array){
			for(double gc : g_array){
				 
				param.C = Math.pow(2, cc);          //以2为Base 进行搜寻
				param.gamma = Math.pow(2, gc) ;
				
				//使用Cross Validation 计算正确率，避免取样偏差
				double accu = CrossValidation(prob, param, 3) ; // problem, parameter, fold
				if(accu > TopAccu){
					
					BetterC = param.C;
					BetterG = param.gamma;
					TopAccu = accu;
					
				}
				
			}
		} // end for
		
		
		
		//将参数设定为最佳参数
	   param.C =  BetterC;
	   param.gamma =BetterG;
		
	   System.out.println("Grid Search Result: C=" +param.C +" Gamma=" + param.gamma);
	   
		return param;
	}

	private static void predict(String TestFileName, String PredictOutputFile, svm_model model)
	{
		
		
		try {
			
			BufferedReader TestFileReader;
			TestFileReader = new BufferedReader(new FileReader(TestFileName));
			DataOutputStream OutStream = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(PredictOutputFile)));
			
			String newline = "\r\n";			
			OutStream.writeBytes("target"+ "\t" + "pridict"+ newline);
		
			
			int correct = 0;
			int total = 0;
			double error = 0;
			double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
			int svm_type=svm.svm_get_svm_type(model);
			int nr_class=svm.svm_get_nr_class(model);
			double[] prob_estimates=null;
			
			//逐行读取测试用档案
			while(true)
			{
				String line = TestFileReader.readLine();
				if(line == null) break;

				StringTokenizer st = new StringTokenizer(line," \t\n\r\f:");

				//取得测试资料 Label , 即为答案
				double target = atof(st.nextToken()); 
				
				int m = st.countTokens()/2;         
				
				svm_node[] x = new svm_node[m];  //特征阵列
				for(int j=0;j<m;j++)
				{
					x[j] = new svm_node();
					x[j].index = atoi(st.nextToken());   //index:value
					x[j].value = atof(st.nextToken());
				}

				double p;   //预测值
				
				
				p = svm.svm_predict(model,x); //输入为model,与特征，输出为预测值。
				
				
				OutStream.writeBytes(target+ "\t" + p + newline); //输出预测结果至档案
				
				
				//进行统计
				if(p == target)
					++correct;
				error += (p-target)*(p-target);
				sumv += p;
				sumy += target;
				sumvv += p*p;
				sumyy += target*target;
				sumvy += p*target;
				++total;
			}
			
			/** 算统计结果 **/
			
			double accuracy = 0.0f ; // 模型预测结果的正确率
			
			if(svm_type == svm_parameter.EPSILON_SVR ||
			   svm_type == svm_parameter.NU_SVR)
			{
				System.out.print("Mean squared error = "+error/total+" (regression)\n");
				System.out.print("Squared correlation coefficient = "+
					 ((total*sumvy-sumv*sumy)*(total*sumvy-sumv*sumy))/
					 ((total*sumvv-sumv*sumv)*(total*sumyy-sumy*sumy))+
					 " (regression)\n");
			}
			else
			{
				accuracy = (double)correct/total;
				
				System.out.print("Accuracy = " + accuracy *100 +
					 "% ("+correct+"/"+total+") (classification)\n");
				
				
			}
			
			OutStream.close();
		
		} catch (FileNotFoundException e) {
			
			e.printStackTrace();
		} catch (IOException e) {
			
			e.printStackTrace();
		} 
    	  
		
		
	}
	
	private static double CrossValidation(svm_problem prob, svm_parameter param, int nr_fold)
	{
		int i;
		int total_correct = 0;
		double total_error = 0;
		double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
		double[] target = new double[prob.l];
         
		svm.svm_cross_validation(prob,param,nr_fold,target);
		if(param.svm_type == svm_parameter.EPSILON_SVR ||
		   param.svm_type == svm_parameter.NU_SVR)
		{
			for(i=0;i<prob.l;i++)
			{
				double y = prob.y[i];
				double v = target[i];
				total_error += (v-y)*(v-y);
				sumv += v;
				sumy += y;
				sumvv += v*v;
				sumyy += y*y;
				sumvy += v*y;
			}
			System.out.print("Cross Validation Mean squared error = "+total_error/prob.l+"\n");
			System.out.print("Cross Validation Squared correlation coefficient = "+
				((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
				((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))+"\n"
				);
		}
		else
		{
			
			for(i=0;i<prob.l;i++){
				if(target[i] == prob.y[i])
					++total_correct;
			}
			
		}
		
		double accu = 100.0*total_correct/prob.l ;
		System.out.print("Cross Validation Accuracy = "+accu+"%\n");
		return accu;
	}
	
	public static svm_parameter SetSVMParameters(double g, double c){
		
		svm_parameter param = new svm_parameter();
		// default values
		param.svm_type = svm_parameter.C_SVC;
		param.kernel_type = svm_parameter.RBF;
		param.degree = 3;
	
		param.coef0 = 0;
		param.nu = 0.5;
		param.cache_size = 100;
		
		param.eps = 1e-3;
		param.p = 0.1;
		param.shrinking = 1;
		param.probability = 0;
		param.nr_weight = 0;
		param.weight_label = new int[0];
		param.weight = new double[0];
		
		param.gamma = g;	// 1/ num_features
		param.C = c;
		
		svm_print_interface print_func = null;	// default printing to stdout
		svm.svm_set_print_string_function(print_func);
		
		return param;
	}
	
	public static svm_problem  CreateProblemFromFile(String trFile) throws IOException{
		
		BufferedReader fp = new BufferedReader(new FileReader(trFile));
		
		Vector<Double> vy = new Vector<Double>();             //Label vector
		Vector<svm_node[]> vx = new Vector<svm_node[]>();     //Feature nodes
		
		int max_index = 0;

		while(true)
		{
			String line = fp.readLine();
			if(line == null || line == "") break;
			
            //拆解每行
			StringTokenizer st = new StringTokenizer(line," \t\n\r\f:"); 

			vy.addElement(atof(st.nextToken()));  //Read Label
			int m = st.countTokens()/2;           //index:value
			svm_node[] x = new svm_node[m];
			
			for(int j=0;j<m;j++)
			{
				x[j] = new svm_node();
				x[j].index = atoi(st.nextToken());  //index:value
				x[j].value = atof(st.nextToken());  
			}
			
			if(m>0) max_index = Math.max(max_index, x[m-1].index);
			vx.addElement(x);                     //Features
		}

		svm_problem prob = new svm_problem();
		prob.l = vy.size();
		
		
		prob.x = new svm_node[prob.l][];
		for(int i=0;i<prob.l;i++)
			prob.x[i] = vx.elementAt(i);
		prob.y = new double[prob.l];
		for(int i=0;i<prob.l;i++)
			prob.y[i] = vy.elementAt(i);


		fp.close();
		
		return prob;
	}
	private static void InsertProblemToDB(svm_problem prob, String IP, int port){
		
		//开启资料库
		try {
					
				MongoClient mongo = new MongoClient(IP, port);
				DB db = mongo.getDB("test");
				DBCollection col = db.getCollection("PersonProfile");
		/*
					 "type" : "rich",
					    "heigh" : 181.1999999999999900,
					    "weight" : 80,
					    "salary" : 20000,
					    "age" : 30
		*/
					 
				
		
		
		for(int i=0;i<prob.l;i++)
		{	
		    
			 BasicDBObject doc = new BasicDBObject();
			 
			 
			 double label = prob.y[i] ; //取得Label
			 String labelStr ="rich" ;
			 
			if(label == 1.0 ){
				labelStr = "rich" ;
			}else if (label == 2.0){
				labelStr = "normal";
			}else if (label >= 3.0){
				labelStr = "poor";
			}else{
				labelStr = "unclassified";
			}
			 
		    doc.append("label", labelStr);
			    
			
			
			
			svm_node[] nodes =prob.x[i].clone();
			
			for(int j = 0 ;j< nodes.length;j++){
				
				int index = nodes[j].index ;
				double value = nodes[j].value;   //index:value		
				switch(index){
				
					case 1 :
						doc.append("heigh", value);
						break;
					case 2 :
						doc.append("weight", value);
						break;
					case 3 :
						doc.append("salary", value);
						break;
					case 4 :
						doc.append("age", value);
						break;
				    default :
				    	doc.append("un", value);
				    	break;
				
				}
			}
			
			 WriteResult result = col.insert(doc);
			
			
		}		// end problem loop	
	
		
	
			 
			 
			 mongo.close();
			
		
		
		
		} catch (UnknownHostException e) {
		
			e.printStackTrace();
		}
		
			
		
	}
	private static svm_problem ReadProblemFromDB(String IP, int port, String DBName){
		
		

		Vector<Double> vy = new Vector<Double>();             //Label vector
		Vector<svm_node[]> vx = new Vector<svm_node[]>();     //Feature nodes
		
		MongoClient mongo;
		try {
			mongo = new MongoClient(IP, port);
			DB db = mongo.getDB("test");
			DBCollection col = db.getCollection("PersonProfile");
			
			BasicDBObject query = new BasicDBObject();
			DBCursor dbc = col.find(query);
			

			
			
			while(dbc.hasNext()){
				
				DBObject dbo = dbc.next() ;
			
				String lbStr = (String) dbo.get("label");
				double lb = -1 ; 
				
				
				double h = (double)dbo.get("heigh");
				double w = (double)dbo.get("weight");
				double   s = (double)dbo.get("salary");
				double  a = (double)dbo.get("age");
				
				
				switch(lbStr){
				
				case "rich" :
					lb = 1;
					break;
				case "normal" :
					lb = 2;
					break;
				case "poor" :
					lb = 3;
					break;
			    default :
			    	lb = -1;
			    	break;
			
				}
				
				vy.addElement(lb);
				
				svm_node[] x = new svm_node[4];
				x[0] = new svm_node();
				x[1] = new svm_node();
				x[2] = new svm_node();
				x[3] = new svm_node();
				
				// index 要从1开始
                x[0].index = 1 ; x[0].value = h;
                x[1].index = 2 ; x[1].value = w;
                x[2].index = 3 ; x[2].value = s;
                x[3].index = 4 ; x[3].value = a;
				vx.addElement(x);    
			}
			
			mongo.close();
		    
			
			
		
		} catch (UnknownHostException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		svm_problem prob = new svm_problem();
		prob.l = vy.size();
		prob.x = new svm_node[prob.l][];
		for(int i=0;i<prob.l;i++)
				prob.x[i] = vx.elementAt(i);
		prob.y = new double[prob.l];
		for(int i=0;i<prob.l;i++)
			prob.y[i] = vy.elementAt(i);
		
		return prob;
	}
	
	private static double atof(String s)
	{
		double d = Double.valueOf(s).doubleValue();
		if (Double.isNaN(d) || Double.isInfinite(d))
		{
			System.err.print("NaN or Infinity in input\n");
			System.exit(1);
		}
		return(d);
	}

	private static int atoi(String s)
	{
		return Integer.parseInt(s);
	}
	
	public static void StoreModeltoFile(svm_model m, String fn) throws IOException{
		String ModelStr = toBase64String((Serializable)m);
		PrintWriter out = new PrintWriter(fn);
		out.println(ModelStr);
		out.close();
		
	}
	
	public static svm_model ReadModelFromFile(String fn) throws IOException, ClassNotFoundException{
		
		
	 BufferedReader in = new BufferedReader(new FileReader(fn));  
	 String Bas64EnStr = in.readLine() ;
	 svm_model model = (svm_model) fromBase64String(Bas64EnStr);	       
		        
		
		return model;
	}

	
    /** Read the object from Base64 string. */
	private static Object fromBase64String( String s ) throws IOException ,
                                                       ClassNotFoundException {
        byte [] data = Base64Coder.decode( s );
        ObjectInputStream ois = new ObjectInputStream( 
                                        new ByteArrayInputStream(data ) );
        Object o  = ois.readObject();
        ois.close();
        return o;
   }

    /** Write the object to a Base64 string. */
    private static String toBase64String( Serializable o ) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream( baos );
        oos.writeObject( o );
        oos.close();
        return new String( Base64Coder.encode( baos.toByteArray() ) );
    }


}
