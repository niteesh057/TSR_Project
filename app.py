from flask import Flask, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64
from io import BytesIO

app = Flask(__name__)

models = {
    'vgg16': load_model('vgg16_classifier_100.keras'),
    'mobilenetv2': load_model('mobilenetv2_classifier_100.keras'),
    'densenet121': load_model('densenet121_classifier_100.keras')
}

class_names = [
    'Accident Prone Area', 'Bumps Ahead', 'Busbay', 'Gas Station',
    'Keep Left', 'Left Chevron', 'No Overtake', 'No Parking',
    'No Stopping', 'Pedestrain Crossing', 'Right Chevron',
    'SchoolZone', 'Seried of Bends', 'Side road Left',
    'Speed brake Ahead', 'SpeedLimit-20 Kmph', 'SpeedLimit-25 Kmph',
    'SpeedLimit-30 Kmph', 'SpeedLimit-40 Kmph', 'SpeedLimit-50 Kmph',
    'SpeedLimit-60 Kmph', 'SpeedLimit-80 Kmph', 'Stop',
    'TrafficLight', 'Turn Left Ahead', 'Turn Right Ahead', 'Uturn'
]

# Placeholder descriptions
descriptions = {
    'Accident Prone Area': {
        'meaning': 'This sign warns drivers of a location where accidents frequently occur. It indicates a need for extra caution due to hazardous conditions.',
        'action': 'Slow down and stay alert for sudden obstacles or other vehicles. Follow any additional speed limits or warnings in the area.'
    },
    'Bumps Ahead': {
        'meaning': 'This sign alerts drivers to speed bumps or uneven road surfaces ahead. It helps prevent damage to vehicles and ensures safety.',
        'action': 'Reduce your speed gradually to avoid jolting your vehicle. Proceed carefully over the bumps to maintain control.'
    },
    'Busbay': {
        'meaning': 'This sign marks a designated area where buses stop to pick up or drop off passengers. It indicates a zone reserved for public transport.',
        'action': 'Avoid parking or stopping in this area unless driving a bus. Yield to buses entering or exiting the bay.'
    },
    'Gas Station': {
        'meaning': 'This sign points to the location of a nearby fuel station. It helps drivers find a place to refuel their vehicles.',
        'action': 'If you need fuel, follow the direction indicated by the sign. Plan your stop and ensure safe entry and exit from the station.'
    },
    'Keep Left': {
        'meaning': 'This sign instructs drivers to stay on the left side of the road or a divider. It ensures proper traffic flow in multi-lane scenarios.',
        'action': 'Move your vehicle to the left side as directed. Do not cross over to the right unless instructed otherwise.'
    },
    'Left Chevron': {
        'meaning': 'This sign highlights a sharp curve or turn to the left on the road. It provides visual guidance for safe navigation.',
        'action': 'Slow down and steer carefully to follow the left curve. Watch for oncoming traffic or obstacles around the turn.'
    },
    'No Overtake': {
        'meaning': 'This sign prohibits overtaking other vehicles on this stretch of road. It is placed where passing could be dangerous.',
        'action': 'Stay behind the vehicle in front and do not attempt to pass. Wait until you see a sign permitting overtaking or a safer section.'
    },
    'No Parking': {
        'meaning': 'This sign indicates that parking is not allowed in the marked area. It helps keep the road clear for traffic flow.',
        'action': 'Do not stop or leave your vehicle unattended in this zone. Find an authorized parking area nearby instead.'
    },
    'No Stopping': {
        'meaning': 'This sign forbids drivers from stopping their vehicles at any time in this area. It ensures continuous movement of traffic.',
        'action': 'Keep driving and do not halt, even briefly, unless in an emergency. Look for a safe spot beyond this zone if you need to stop.'
    },
    'Pedestrain Crossing': {
        'meaning': 'This sign warns of a designated crossing area for pedestrians ahead. It alerts drivers to watch for people on foot.',
        'action': 'Slow down and be prepared to stop if pedestrians are crossing. Yield the right of way to ensure their safety.'
    },
    'Right Chevron': {
        'meaning': 'This sign marks a sharp curve or turn to the right on the road. It guides drivers to navigate the turn safely.',
        'action': 'Reduce your speed and steer carefully to follow the right curve. Stay vigilant for traffic or hazards around the bend.'
    },
    'SchoolZone': {
        'meaning': 'This sign indicates an area near a school where children may be present. It calls for heightened awareness due to pedestrian activity.',
        'action': 'Drive slowly and watch for children crossing or walking nearby. Obey any posted speed limits or crossing guards.'
    },
    'Seried of Bends': {
        'meaning': 'This sign warns of multiple curves or bends in the road ahead. It prepares drivers for a winding section that requires caution.',
        'action': 'Lower your speed and maintain steady control through each bend. Avoid sudden maneuvers to stay safe on the curves.'
    },
    'Side road Left': {
        'meaning': 'This sign indicates a side road joining from the left. It alerts drivers to potential traffic merging into the main road.',
        'action': 'Be cautious of vehicles entering from the left side road. Adjust your speed or position to allow safe merging if needed.'
    },
    'Speed brake Ahead': {
        'meaning': 'This sign signals the presence of a speed breaker or bump ahead on the road. It warns drivers to reduce speed for safety.',
        'action': 'Slow down well in advance to cross the speed breaker smoothly. Proceed carefully to avoid discomfort or vehicle damage.'
    },
    'SpeedLimit-20 Kmph': {
        'meaning': 'This sign sets a maximum speed limit of 20 kilometers per hour in this area. It ensures safety in low-speed zones.',
        'action': 'Adjust your speed to stay at or below 20 kmph. Check your speedometer regularly to comply with the limit.'
    },
    'SpeedLimit-25 Kmph': {
        'meaning': 'This sign establishes a maximum speed of 25 kilometers per hour for this section. It is often used in residential or busy areas.',
        'action': 'Keep your speed under 25 kmph to follow the restriction. Be prepared for pedestrians or traffic requiring a slower pace.'
    },
    'SpeedLimit-30 Kmph': {
        'meaning': 'This sign indicates a maximum speed limit of 30 kilometers per hour on this road. It promotes safe driving in moderate-traffic zones.',
        'action': 'Maintain your speed at or below 30 kmph as required. Watch for signs lifting the limit before increasing speed.'
    },
    'SpeedLimit-40 Kmph': {
        'meaning': 'This sign sets the maximum speed to 40 kilometers per hour in this area. It balances safety and efficiency on the road.',
        'action': 'Do not exceed 40 kmph while driving through this zone. Stay alert for changes in traffic conditions or signage.'
    },
    'SpeedLimit-50 Kmph': {
        'meaning': 'This sign restricts the maximum speed to 50 kilometers per hour on this stretch. It is common in semi-urban or controlled areas.',
        'action': 'Keep your speed at or below 50 kmph to stay compliant. Reduce speed if conditions like weather or traffic worsen.'
    },
    'SpeedLimit-60 Kmph': {
        'meaning': 'This sign allows a maximum speed of 60 kilometers per hour in this section. It indicates a relatively open or safe road.',
        'action': 'Drive at or below 60 kmph, adhering to the limit. Remain cautious of other vehicles or road changes ahead.'
    },
    'SpeedLimit-80 Kmph': {
        'meaning': 'This sign permits a maximum speed of 80 kilometers per hour on this road. It is typically used on highways or major routes.',
        'action': 'Maintain a speed of 80 kmph or less as allowed. Ensure your vehicle is in good condition for higher speeds.'
    },
    'Stop': {
        'meaning': 'This sign mandates a complete stop at the designated point on the road. It ensures safety at intersections or crossings.',
        'action': 'Bring your vehicle to a full stop and check all directions for traffic. Proceed only when it is safe and clear to do so.'
    },
    'TrafficLight': {
        'meaning': 'This sign indicates the presence of a traffic light ahead controlling the flow of vehicles. It prepares drivers for signal changes.',
        'action': 'Approach the traffic light cautiously and obey its signals. Stop on red, prepare on yellow, and go on green as directed.'
    },
    'Turn Left Ahead': {
        'meaning': 'This sign warns of a mandatory left turn upcoming on the road. It guides drivers to prepare for the maneuver.',
        'action': 'Signal early and position your vehicle for a left turn. Complete the turn carefully, yielding to oncoming traffic if needed.'
    },
    'Turn Right Ahead': {
        'meaning': 'This sign indicates a mandatory right turn ahead on the route. It helps drivers anticipate the change in direction.',
        'action': 'Use your right indicator and align your vehicle for the turn. Execute the right turn safely, checking for other traffic.'
    },
    'Uturn': {
        'meaning': 'This sign permits or directs drivers to make a U-turn at this point. It allows a change in direction on the same road.',
        'action': 'Signal your intent and check for oncoming traffic before making the U-turn. Perform the turn smoothly and safely within the allowed space.'
    },
    'default': {
        'meaning': 'No Traffic Sign Detected.',
        'action': 'Nothing to do.'
    }
}

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        file_path = 'temp.jpg'
        file.save(file_path)
        
        # Preprocess the image for prediction
        img = cv2.imread(file_path)
        img_resized = cv2.resize(img, (64, 64))
        img_normalized = img_resized / 255.0
        img_input = np.expand_dims(img_normalized, axis=0)
        
        # Predict with hybrid model (weighted: 25% VGG16, 50% MobileNetV2, 25% DenseNet121)
        predictions = {name: model.predict(img_input) for name, model in models.items()}
        hybrid_pred = (0.25 * predictions['vgg16'] + 0.50 * predictions['mobilenetv2'] + 0.25 * predictions['densenet121'])
        predicted_class_idx = np.argmax(hybrid_pred)
        predicted_class = class_names[predicted_class_idx]
        confidence = hybrid_pred[0][predicted_class_idx]
        
        # Get description
        info = descriptions.get(predicted_class, descriptions['default'])
        
        # Convert image to base64 for display
        _, img_encoded = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        
        return jsonify({
            'prediction': predicted_class,
            'meaning': info['meaning'],
            'action': info['action'],
            'image': img_base64
        })
    
    # Serve the HTML page for GET requests
    with open('index.html', 'r') as f:
        return f.read()

if __name__ == '__main__':
    app.run(debug=True)