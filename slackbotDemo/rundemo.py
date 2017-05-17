from squad.demo_prepro import prepro
from basic.demo_cli import Demo

class BidafBot:
        
    def __init__(self):
        self.paragraph = None
        self.question = None
        self.demo = Demo()
	
    
    def getAnswer(self, paragraph, question):

        pq_prepro = prepro(paragraph, question)
        if len(pq_prepro['x'])>1000:
            return "[Error] Sorry, the number of words in paragraph cannot be more than 1000." 
        if len(pq_prepro['q'])>100:
            return "[Error] Sorry, the number of words in question cannot be more than 100."
        return self.demo.run(pq_prepro)

    def respond(self,query):
        if query.startswith("@context"):
           self.paragraph = query.split("@context")[1]
           answer = "Context received. Shoot your questions"
        else :
            self.question = query
            answer = self.getAnswer(self.paragraph,self.question)
        return answer


