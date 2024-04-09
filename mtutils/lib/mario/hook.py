

class Hook:

    def __init__(self) -> None:
        pass

    def before_run(self, pipeline, databus):
        pass

    def after_run(self, pipeline, databus):
        pass

    def before_global_infer(self, pipeline, databus_list):
        pass

    def after_global_infer(self, pipeline, databus_list):
        pass

    def before_single_infer(self, pipeline, databus):
        pass

    def after_single_infer(self, pipeline, databus):
        pass
