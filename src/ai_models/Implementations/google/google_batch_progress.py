from vertexai.preview.batch_prediction import BatchPredictionJob


class GoogleBatchManager:
    def __init__(self):
        pass

    def get_batch_info(self, batch_id):
        try:
            return BatchPredictionJob(batch_id)
        except Exception as e:
            print(f"Error retrieving batch info for {batch_id}: {e}")
            return None

    def display_batch_progress(self, batch_info):
        if not batch_info:
            print(f"No batch info available")
            return

        try:
            if batch_info.has_ended:
                if batch_info.has_succeeded:
                    print(f"Batch ID: {batch_info.resource_name} - fully completed")
                else:
                    print(f"Batch ID: {batch_info.resource_name} - failed")
            else:
                print(f"Batch ID: {batch_info.resource_name} - running")
            print("-" * 60)
        except Exception as e:
            print(f"Error displaying batch progress: {e}")
            print(f"Batch ID: {getattr(batch_info, 'resource_name', 'unknown')}")
            print("Unable to display progress information")
            print("-" * 60)

    def show_batch_progress(self, batch_ids):
        for batch_id in batch_ids:
            batch_info = self.get_batch_info(batch_id)
            if batch_info:
                self.display_batch_progress(batch_info)
            else:
                print(f"No information available for batch {batch_id}")
                print("-" * 60)
