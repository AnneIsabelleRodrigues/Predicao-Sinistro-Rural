import ee

class GoogleEarthEngine():
    def __init__(self):
        self.service_account = 'poc-sinistro-rural@poc-1-397804.iam.gserviceaccount.com'
        self.key = 'keys/poc-1-397804-454e15b1a666.json'

    def initilize(self):
        credentials = ee.ServiceAccountCredentials(self.service_account,self.key)
        ee.Initialize(credentials)