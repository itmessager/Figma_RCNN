class ReusableNN:
    def __init__(self, net):
        self.net = net
        self.last_data_id = None
        self.last_results = None

    def __call__(self, data_id, *args):
        if self.last_data_id == data_id:
            return self.last_results

        self.last_data_id = data_id
        self.last_results = self.net(args)
        return self.last_results
