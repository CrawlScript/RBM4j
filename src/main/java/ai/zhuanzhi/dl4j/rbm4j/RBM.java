package ai.zhuanzhi.dl4j.rbm4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;


/**
 * @author hu
 * 该教程由专知（www.zhuanzhi.ai）提供
 */
public class RBM {

    protected INDArray w, b, a;

    public RBM(int vDim, int hDim){
        w = Nd4j.rand(vDim, hDim);
        b = Nd4j.rand(new int[]{hDim});
        a = Nd4j.rand(new int[]{vDim});
    }

    //输入可见层v，输出隐藏层为1的概率p(h|v)
    protected INDArray computeHProbs(INDArray vSamples){
        INDArray hProbs = vSamples.mmul(w).addRowVector(b);
        hProbs = Transforms.sigmoid(hProbs);
        return hProbs;
    }

    //输入隐藏层h，输出可见层为1的概率p(v|h)
    protected INDArray computeVProbs(INDArray hSamples){
        INDArray vProbs = hSamples.mmul(w.transpose()).addRowVector(a);
        vProbs = Transforms.sigmoid(vProbs);
        return vProbs;
    }

    //输入可见层，输出RBM重构的结果
    public INDArray reconstruct(INDArray vSamples){
        INDArray hProbs = computeHProbs(vSamples);
        INDArray hSamples = bernoulliSample(hProbs);
        INDArray negVSamples = bernoulliSample(computeVProbs(hSamples));
        return negVSamples;
    }

    public double fit(INDArray vSamples, double learningRate){
        INDArray hProbs = computeHProbs(vSamples);
        INDArray hSamples = bernoulliSample(hProbs);
        INDArray negVSamples = bernoulliSample(computeVProbs(hSamples));
        INDArray negHProbs = computeHProbs(negVSamples);

        INDArray mseTempMatrix = negVSamples.sub(vSamples);
        double loss = mseTempMatrix.mul(mseTempMatrix).div(2).mean(0,1).getDouble(0,0);

        //正梯度
        INDArray posGrad = vSamples.transpose().mmul(hProbs);
        //负梯度
        INDArray negGrad = negVSamples.transpose().mmul(negHProbs);
        //输入样本数量
        int numSamples = vSamples.shape()[0];
        //计算并更新参数
        INDArray dw = posGrad.sub(negGrad).mul(learningRate/numSamples);
        INDArray db = hProbs.mean(0).sub(negHProbs.mean(0)).mul(learningRate);
        INDArray da = vSamples.mean(0).sub(negVSamples.mean(0)).mul(learningRate);
        w.addi(dw);
        b.addi(db);
        a.addi(da);

        return loss;
    }

    //伯努利采样
    protected INDArray bernoulliSample(INDArray probs){
        INDArray randArray = Nd4j.rand(probs.shape());
        INDArray samples = probs.gt(randArray);
        return samples;
    }

    public static void main(String[] args) {
        //手工设置一组数据
        double[][] rawVSamples = new double[][]{
                {1,1,1,1,0,0,0,0},
                {1,1,1,1,0,0,0,0},
                {0,0,1,1,1,1,0,0},
                {0,0,1,1,1,1,0,0},
                {0,0,0,0,1,1,1,1},
                {0,0,0,0,1,1,1,1}
        };
        INDArray vSamples = Nd4j.create(rawVSamples);

        //设置RBM（隐藏层大小为2）
        RBM rbm = new RBM(vSamples.shape()[1],2);
        //训练
        for(int i=0;i<20000;i++){
            double loss = rbm.fit(vSamples,5e-3);
            if(i % 1000 == 0){
                System.out.println("batch:"+i+"\tloss:" + loss);
            }
        }
        //显示重构结果
        System.out.println("reconstruction:");
        System.out.println(rbm.reconstruct(vSamples));
        //显示对应的隐藏层激活值（学习到的特征）
        System.out.println("features:");
        System.out.println(rbm.computeHProbs(vSamples));

    }
}
