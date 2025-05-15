import SwiftUI

struct ChatView: View {
    @EnvironmentObject var chatbot: Chatbot
    @FocusState private var isInputFocused: Bool

    var body: some View {
        VStack {
            ScrollViewReader { proxy in //pour scroller
                ScrollView {
                    VStack(spacing: 8) {
                        ForEach(chatbot.messages) { msg in
                            MessageBubble(message: msg)
                                .id(msg.id)
                        }
                    }
                    .padding(.vertical, 10)
                    .padding(.horizontal)
                    .contentShape(Rectangle())
                    .simultaneousGesture(
                        TapGesture().onEnded {
                            isInputFocused = false
                        }
                    )
                    .simultaneousGesture(
                        DragGesture().onChanged { _ in
                            isInputFocused = false
                        }
                    )
                }
                .onChange(of: chatbot.messages.count) {
                    scrollToLast(proxy) //scroll automatiquement
                }
                .background(Color(.systemGray6))
            }

            Divider()

            HStack(spacing: 12) {
                TextField("Type a message", text: $chatbot.userInput)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .focused($isInputFocused)
                    .onSubmit(sendIfValid)

                Button(action: sendIfValid) {
                    Image(systemName: "paperplane.fill")
                        .foregroundColor(.white)
                        .padding(10)
                        .background(Color.blue)
                        .clipShape(Circle())
                }
                .disabled(chatbot.userInput.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            .padding()
            .background(Color(.systemBackground))
        }
        .navigationTitle("COREX Chatbot")
        .background(Color(.systemGray6))
    }

    private func sendIfValid() {
        let input = chatbot.userInput.trimmingCharacters(in: .whitespaces)
        if !input.isEmpty {
            chatbot.sendMessage() //sassure que c pas vide le message
            
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                isInputFocused = true
            }
        }
    }

    private func scrollToLast(_ proxy: ScrollViewProxy) {
        if let last = chatbot.messages.last {
            withAnimation(.easeOut(duration: 0.3)) {
                proxy.scrollTo(last.id, anchor: .bottom) //defile auto
            }
        }
    }
}

struct ChatView_Previews: PreviewProvider {
    static var previews: some View {
        ChatView().environmentObject(Chatbot())
    }
}

//need to rework it to me it ours
