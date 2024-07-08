import unittest
from flask import json
from src.app import app

class TestToSerApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True 

    def test_index_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'ToSer', response.data)

    def test_analyze_route_no_url(self):
        response = self.app.post('/analyze', 
                                 data=json.dumps({}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'No URL provided')

    def test_analyze_route_invalid_url(self):
        response = self.app.post('/analyze', 
                                 data=json.dumps({'url': 'http://invalidurl.com'}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('Unable to fetch the Terms of Service document', data['error'])

    def test_analyze_route_valid_url(self):
        # This test assumes that the URL will return a valid ToS document
        # You might want to mock the external API calls for more reliable testing
        response = self.app.post('/analyze', 
                                 data=json.dumps({'url': 'https://www.example.com/tos'}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('categories', data)
        self.assertIn('overall_score', data)
        self.assertIn('summary', data)

    def test_404_error(self):
        response = self.app.get('/nonexistent-route')
        self.assertEqual(response.status_code, 404)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'Not found')

if __name__ == '__main__':
    unittest.main()
