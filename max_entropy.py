from classifier import Classifier
import numpy

import scipy.optimize
def Main():
    a=numpy.array([[1,2,3],[1,1,1]])
    print numpy.sum(a, 0)

class MaxEntropy(Classifier):
    def __init__(self, model={}):
        super(MaxEntropy, self).__init__(model)
        self.data = {}
        self.label_y={}
        self.feature_dictionary ={}
        self.feature_x=numpy.array([])
        self.observedE = numpy.array([])
        self.expectedE_weigh=numpy.array([])
        self.pos_prob=numpy.array([])
        self.blog_pos_prob=numpy.array([])
        self.lam=numpy.array([])
        self.DELTA=2
        self.feature_section1=numpy.array([[]])
        self.feature_section2=numpy.array([[]])
        self.feature_section3=numpy.array([[]])
        self.feature_section4=numpy.array([[]])
        self.feature_section5=numpy.array([[]])
    
    def get_model(self): return self.seen
    def set_model(self, model): self.seen = model
    model = property(get_model, set_model)
    
    def _init_observedE(self):
        """Calculate the observed expectation of given (y,x), and this could be occurrence number of each (y,x) when calculating the likelihood"""
        for y in self.label_y.iterkeys():
            self.observedE=numpy.append(self.observedE,self.label_y[y])
            for i in range(len(self.feature_x)):
                x=self.feature_x[i][0]
                if self.data.has_key((y,x)):
                    self.observedE=numpy.append(self.observedE, self.data.get((y,x)))
                else:
                    self.observedE=numpy.append(self.observedE, 0)
        
    def _init_expected_weigh(self):
        """Calculate the weigh of each expected expectation of given (y,x)"""
        self.expectedE_weigh=numpy.reshape(self.observedE,(2,self.observedE.size))
        self.expectedE_weigh=numpy.sum(self.expectedE_weigh, axis=0)
        self.expectedE_weigh=numpy.append(self.expectedE_weigh, self.expectedE_weigh)
        
    def cal_log_likelihood(self,lam):
        """Calculate the log likelihood"""
        lam=numpy.reshape(lam, (len(self.label_y),len(self.feature_x)+1))
        regularization=sum((lam**2)/(self.DELTA**2))
        for y in range(len(self.label_y)):
            for feature_ocur in self.feature_section1:
                if y==feature_ocur[0]:
                    feature_ocur[0]=1
                    pos=numpy.exp(numpy.dot(lam[y],feature_ocur)) / numpy.sum((numpy.exp(numpy.dot(lam[i],feature_ocur))) for i in range(len(self.label_y)))
                    self.blog_pos_prob=numpy.append(self.blog_pos_prob, pos)
        self.blog_pos_prob=numpy.log(self.blog_pos_prob)
        result=numpy.sum(self.blog_pos_prob)+regularization
        return -result
    
    def cal_pos_prob(self,lam):
        """calculate the prosperior probability"""
        self.pos_prob=numpy.array([])
        lam=numpy.reshape(lam, (len(self.label_y),len(self.feature_x)+1))
        for y in range(len(self.label_y)):
            pos=numpy.exp(lam[y]) / sum((numpy.exp(lam[i])) for i in range(len(self.label_y)))
            self.pos_prob=numpy.append(self.pos_prob, pos)
        
    def gradient(self,lam):
        """Calculate the gradient"""
        self.cal_pos_prob(lam)
        regularization=(2*lam)/(self.DELTA**2)
        grad=self.observedE-numpy.log(self.pos_prob*self.expectedE_weigh)-regularization
        return -grad
                        
                        
    def _init_feature(self,instances):
        flag=0
#        for instance in instances:
#            tmp_feature_section=numpy.zeros(len(self.feature_x)+1)
#            features=instance.features()
#            for feature in features:
#                item_index=self.feature_dictionary.get(feature)+1
#                tmp_feature_section[item_index]=1
#            self.feature_section1=numpy.append(self.feature_section1, tmp_feature_section)
#            flag=flag+1
#        self.feature_section1=numpy.reshape(self.feature_section1, (30,(len(self.feature_x)+1)))
    
        for instance in instances:
            tmp_feature_section=numpy.zeros(len(self.feature_x)+1)
            self.feature_section1[0]=1
            features=instance.features()
            if instance.label=='M':
                tmp_feature_section[0]=0
            else:
                tmp_feature_section[0]=1
            for feature in features:
                item_index=self.feature_dictionary.get(feature)+1
                tmp_feature_section[item_index]=1
            if flag<=599:
                self.feature_section1=numpy.append(self.feature_section1, tmp_feature_section)
            elif flag>599 and flag<=1199:
                self.feature_section2=numpy.append(self.feature_section2, tmp_feature_section)
            elif flag>1199 and flag<=1799:
                self.feature_section3=numpy.append(self.feature_section3, tmp_feature_section)
            elif flag>1799 and flag<=2399:
                self.feature_section4=numpy.append(self.feature_section4, tmp_feature_section)
            else:
                self.feature_section5=numpy.append(self.feature_section5, tmp_feature_section)
            flag=flag+1
        self.feature_section1=numpy.reshape(self.feature_section1, (600,(len(self.feature_x)+1)))
        self.feature_section2=numpy.reshape(self.feature_section2, (600,(len(self.feature_x)+1)))
        self.feature_section3=numpy.reshape(self.feature_section3, (600,(len(self.feature_x)+1)))
        self.feature_section4=numpy.reshape(self.feature_section4, (600,(len(self.feature_x)+1)))
        self.feature_section5=numpy.reshape(self.feature_section5, (600,(len(self.feature_x)+1)))
        self.feature_section1=numpy.vstack((self.feature_section1,self.feature_section2,self.feature_section3,self.feature_section4,self.feature_section5))
        self.observedE=numpy.sum(self.feature_section1,0)
    
    
    def __initparams(self,instances):
        """Initiate parameters, record the information from training set"""
        order_number=0
        for instance in instances:
            if instance.label in self.label_y:
                self.label_y[instance.label]+=1
            else:
                self.label_y[instance.label]=1
            features=instance.features()
            for feature in features:
                if self.data.has_key((instance.label,feature)):
                    self.data[(instance.label,feature)]+=1
                else:
                    self.data[(instance.label,feature)]=1
                if feature in self.feature_dictionary:
                    pass
                else:
                    self.feature_dictionary[feature]=order_number
                    order_number=order_number+1
        self.feature_x= sorted(self.feature_dictionary.iteritems(), key=lambda d:d[1])
        self._init_feature(instances)
#        self._init_observedE()   
        self._init_expected_weigh()
        
    def train(self, instances):
        """Train method"""
        self.__initparams(instances)
        size=len(self.feature_x)*len(self.label_y)+2
        i=scipy.optimize.fmin_l_bfgs_b(self.cal_log_likelihood,numpy.zeros(size) , fprime=self.gradient)
        self.model['lam']=i[0]


    def classify(self, instance):
        """Classify method"""
        lam=self.model['lam']
        lam=numpy.reshape(lam, (len(self.label_y),len(self.feature_x)+1))
        prob_list=numpy.array([])
        feature_ocur=numpy.zeros(len(self.feature_x)+1)
        for feature in instance.features():
            if feature in self.feature_dictionary:
                item_index=self.feature_dictionary.get(feature)+1
                feature_ocur[item_index]=1
            else:
                pass
            
        for y in range(len(self.label_y)):
            feature_ocur[0]=1
            pos=numpy.exp(numpy.dot(lam[y],feature_ocur)) / numpy.sum((numpy.exp(numpy.dot(lam[i],feature_ocur))) for i in range(len(self.label_y)))
            prob_list=numpy.append(pos, pos)

        prob_M=prob_list[0]
        prob_F=prob_list[1]
        if prob_M<prob_F:
            return 'F'
        else:
            return 'M'
        
    if __name__ == '__main__':
        Main()