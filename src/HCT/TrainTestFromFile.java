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
import java.util.StringTokenizer;
import java.util.Vector;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_print_interface;
import libsvm.svm_problem;

public class TrainTestFromFile {

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
			
			//double accu = CrossValidation(prob, param, 3) ; // problem, parameter, fold
			
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
				//svm.svm_predict_probability(arg0, arg1, arg2); output probability of all classes
				//svm.svm_predict_values(arg0, arg1, arg2); //Regression
				
				OutStream.writeBytes(target+ "\t" + p + newline); //输出预测结果至档案
				double[] pa = new double[4];
				
				
				System.out.println(target+ "\t" + p );
				double a = svm.svm_predict_probability(model,x,pa);
				for(double v : pa){
					
				}
				
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
		String ModelStr = toString((Serializable)m);
		PrintWriter out = new PrintWriter(fn);
		out.println(ModelStr);
		out.close();
		
	}
	
	public static svm_model ReadModelFromFile(String fn) throws IOException, ClassNotFoundException{
		
		
	 BufferedReader in = new BufferedReader(new FileReader(fn));  
	 String Bas64EnStr = in.readLine() ;
	 svm_model model = (svm_model) fromString(Bas64EnStr);	       
		        
		
		return model;
	}

	
    /** Read the object from Base64 string. */
	private static Object fromString( String s ) throws IOException ,
                                                       ClassNotFoundException {
        byte [] data = Base64Coder.decode( s );
        ObjectInputStream ois = new ObjectInputStream( 
                                        new ByteArrayInputStream(  data ) );
        Object o  = ois.readObject();
        ois.close();
        return o;
   }

    /** Write the object to a Base64 string. */
    private static String toString( Serializable o ) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream( baos );
        oos.writeObject( o );
        oos.close();
        return new String( Base64Coder.encode( baos.toByteArray() ) );
    }


}
