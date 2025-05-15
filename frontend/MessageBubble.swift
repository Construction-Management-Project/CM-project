import SwiftUI

struct MessageBubble: View {
    let message: ChatMessage

    var body: some View {
        
        HStack { //de gauche a droite cette fois
            if message.isFromUser {
                Spacer() //met le message a droite
                Text(message.text)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(16)
                    .frame(maxWidth: 250, alignment: .trailing) //le trailing le met bien a droite
            } else {
                Text(message.text)
                    .padding()
                    .background(Color.pink)
                    .foregroundColor(.black)
                    .cornerRadius(16)
                    .frame(maxWidth: 250, alignment: .leading)
                Spacer()
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 2) //pas sur de lutilite 
    }
}
